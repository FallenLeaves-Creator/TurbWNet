import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
import math
from torchstat import stat
import torch.nn.functional as tf
import torchvision.transforms as transforms
scharr_x = torch.tensor([
    [-3, 0, 3],
    [-10, 0, 10],
    [-3, 0, 3]]).float().unsqueeze(0).unsqueeze(0).to(device='cuda')

scharr_y = torch.tensor([
    [-3, -10, -3],
    [0, 0, 0],
    [3, 10, 3]]).float().unsqueeze(0).unsqueeze(0).to(device='cuda')
def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Overlapping Cross-Attention (OCA)
class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))

        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads, h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads, spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()


        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class SR_Upsample(nn.Sequential):
    """SR_Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of features.
    """

    def __init__(self, scale, num_feat):
        m = []

        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, kernel_size = 3, stride = 1, padding = 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(SR_Upsample, self).__init__(*m)

##########################################################################

@ARCH_REGISTRY.register()
class XRestormer(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        channel_heads = [1,2,4,8],
        spatial_heads = [2,2,3,4],
        overlap_ratio=[0.5, 0.5, 0.5, 0.5],
        window_size = 8,
        spatial_dim_head = 16,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        scale = 1
    ):

        super(XRestormer, self).__init__()
        print("Initializing XRestormer")
        self.scale = scale

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), window_size = window_size, overlap_ratio=overlap_ratio[3],  num_channel_heads=channel_heads[3], num_spatial_heads=spatial_heads[3], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), window_size = window_size, overlap_ratio=overlap_ratio[2],  num_channel_heads=channel_heads[2], num_spatial_heads=spatial_heads[2], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[1],  num_channel_heads=channel_heads[1], num_spatial_heads=spatial_heads[1], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        if self.scale > 1:
            inp_img = F.interpolate(inp_img, scale_factor=self.scale, mode='bilinear', align_corners=False)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1


class HookTool:
    def __init__(self):
        self.fea = None

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out

@ARCH_REGISTRY.register()
class Wnet(nn.Module):
    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out
    def __init__(self,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 4,
        channel_heads = [1,2,4,8],
        spatial_heads = [2,2,3,4],
        overlap_ratio=[0.5, 0.5, 0.5, 0.5],
        window_size = 8,
        spatial_dim_head = 16,
        bias = False,
        ffn_expansion_factor = 2.66,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        scale = 1,
        deturb_path=False,
        detilt_path=False,
    ):

        super(Wnet, self).__init__()
        print("Initializing Wnet")
        self.deblur_model=XRestormer(inp_channels=3,
        out_channels=3,
        dim=dim,
        num_blocks = num_blocks,
        num_refinement_blocks = num_refinement_blocks,
        channel_heads = channel_heads,
        spatial_heads = spatial_heads,
        overlap_ratio= overlap_ratio,
        window_size = window_size,
        spatial_dim_head = spatial_dim_head,
        bias = bias,
        ffn_expansion_factor = ffn_expansion_factor,
        LayerNorm_type = LayerNorm_type,
        scale = scale)

        if deturb_path:
            self.deblur_model.load_state_dict(
                torch.load(deturb_path)["params"]
            )

        for n, m in self.deblur_model.named_modules():
            if n.endswith("refinement"):
                self.deblur_Hook = HookTool()
                m.register_forward_hook(self.deblur_Hook.hook_fun)

        # self.deblur_model.requires_grad=False

        self.detilt_model=XRestormer( inp_channels=2,
        out_channels=2,
        dim=dim,
        num_blocks = num_blocks,
        num_refinement_blocks = num_refinement_blocks,
        channel_heads = channel_heads,
        spatial_heads = spatial_heads,
        overlap_ratio= overlap_ratio,
        window_size = window_size,
        spatial_dim_head = spatial_dim_head,
        bias = bias,
        ffn_expansion_factor = ffn_expansion_factor,
        LayerNorm_type = LayerNorm_type,
        scale = scale)

        if detilt_path:
            self.detilt_model.load_state_dict(
                torch.load(detilt_path)["params"]
            )

        # self.deblur_model.requires_grad=False
        refinement=[]
        for i in range(num_refinement_blocks):
            refinement.append(TransformerBlock(dim=dim*2**1, window_size = window_size, overlap_ratio=overlap_ratio[0],  num_channel_heads=channel_heads[0], num_spatial_heads=spatial_heads[0], spatial_dim_head = spatial_dim_head, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type))
        refinement.append(nn.Conv2d(dim*2**1,3,kernel_size=3, stride=1, padding=1, bias=bias))
        self.final_refine = nn.Sequential(*refinement)
        # self.final_refine=nn.Sequential(
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(int(dim/2),3,kernel_size=3, stride=1, padding=1, bias=bias)
        # )


    def flow_warp(self,
                x,
                flow,
                interpolation='bilinear',
                padding_mode='border',
                align_corners=True):
        """Warp an image or a feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
                a two-channel, denoting the width and height relative offsets.
                Note that the values are not normalized to [-1, 1].
            interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
                Default: 'bilinear'.
            padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Whether align corners. Default: True.
        Returns:
            Tensor: Warped image or feature map.
        """
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                            f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)  # n * h * w * 2
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def forward(self, inp_img):
        deblur_result=self.deblur_model(inp_img)
        gray_img= transforms.Grayscale()(deblur_result)
        gray_img_x_grad=tf.conv2d(gray_img,scharr_x,padding='same')
        gray_img_y_grad=tf.conv2d(gray_img,scharr_y,padding='same')
        grad_deblur_result=torch.cat((gray_img_x_grad,gray_img_y_grad),dim=1)
        detilt_flow=self.detilt_model(grad_deblur_result)
        # output=self.flow_warp(deblur_result,detilt_flow.permute(0,2,3,1))
        output=self.final_refine(self.flow_warp(self.deblur_Hook.fea,detilt_flow.permute(0,2,3,1)))
        return output