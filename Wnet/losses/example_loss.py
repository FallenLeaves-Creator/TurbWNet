from torch import nn as nn
from torch.nn import functional as F
import torch
from basicsr.utils.registry import LOSS_REGISTRY
from torch.autograd import Variable
from math import exp
import lpips
from torchvision.transforms.functional import normalize
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
import numpy as np
import os
from torch.nn import MSELoss, L1Loss

@LOSS_REGISTRY.register()
class ExampleLoss(nn.Module):
    """Example Loss.

    Args:
        loss_weight (float): Loss weight for Example loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(ExampleLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * F.l1_loss(pred, target, reduction='mean')


@LOSS_REGISTRY.register()
class CostVolunmLoss(nn.Module):


    def __init__(self):
        super(CostVolunmLoss, self).__init__()


    def forward(self, pred, target, **kwargs):

        return torch.corrcoef()


def type_trans(window,img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mcs_map  = (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)
        ssim_map, mcs_map =_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map

@LOSS_REGISTRY.register()
class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11,size_average = True):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # self.channel = 3

    def forward(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

        msssim = Variable(torch.Tensor(levels,))
        mcs    = Variable(torch.Tensor(levels,))

        if torch.cuda.is_available():
            weight =weight.cuda()
            msssim=msssim.cuda()
            mcs=mcs.cuda()

        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)

        for i in range(levels): #5 levels
            ssim_map, mcs_map = _ssim(img1, img2,window,self.window_size, channel, self.size_average)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            # print(img1.shape)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1 #refresh img
            img2 = filtered_im2

        return torch.prod((msssim[levels-1]**weight[levels-1] * mcs[0:levels-1]**weight[0:levels-1]))

@LOSS_REGISTRY.register()
class Lpips_loss(torch.nn.Module):
    def __init__(self, loss_weight = 1.0,net = 'alex'):
        super(Lpips_loss, self).__init__()
        self.weight = loss_weight
        self.loss_fn = lpips.LPIPS(net=net)
        self.loss_fn.requires_grad_=False

    def forward(self, pred,target):



        return self.weight*self.loss_fn(pred,target).mean()

#高斯滤波器的正态分布，window_size是位置参数，决定分布的位置；sigma是尺度参数，决定分布的幅度。
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])  #双星号：幂的意思。 双//：表示向下取整，有一方是float型时，结果为float。  exp()返回e的x次方。
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1) #unsqueeze（x）增加维度x
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)  #t() 将tensor进行转置。  x.mm(self.y) 将x与y相乘。
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

#返回的均值。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    #求像素的动态范围
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    #求img1，img2的均值。
    padd = 0
    (_, channel, height, width) = img1.size() # _ 为批次batch大小。
        #定义卷积核window
    if window is None:
        real_size = min(window_size, height, width) #求最小值，是为了保证卷积核尺寸和img1，img2尺寸相同。
        window = create_window(real_size, channel=channel).to(img1.device)

        #空洞卷积：有groups代表是空洞卷积；  F.conv2d(输入图像tensor，卷积核tensor, ...)是卷积操作。
        #mu1为img1的均值；mu2为img2的均值。
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel) #groups控制分组卷积，默认不分组，即为1.  delition默认为1.
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel) #conv2d输出的是一个tensor-新的feature map。

        #mu1_sq:img1均值的平方。 mu2_sq:img2均值的平方
    mu1_sq = mu1.pow(2) #对mu1中的元素逐个2次幂计算。
    mu2_sq = mu2.pow(2)
        #img1,img2均值的乘积。
    mu1_mu2 = mu1 * mu2

    #x的方差σx²
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    #y的方差σy²
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    #求x,y的协方差σxy
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    #维持稳定的两个变量
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    #v1:2σxy+C2
    v1 = 2.0 * sigma12 + C2
    #v2:σx²+σy²+C2
    v2 = sigma1_sq + sigma2_sq + C2
    assert v2
    cs = torch.mean(v1 / v2)  # contrast sensitivity   #对比敏感度

    #ssim_map为img1,img2的相似性指数。
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    #求平均相似性指数。 ??
    if size_average: #要求平均时
        ret = ssim_map.mean()
    else: #不要求平均时
        ret = ssim_map.mean(1).mean(1).mean(1) #mean(1) 求维度1的平均值

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device) #to(device)使用GPU运算
    # weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    mssim = [] #存放每一尺度的ssim的平均值
    mcs = [] #存放每一尺度的cs的平均值
    #将img1，img2两张图像分为levels个小窗口，求每对小窗口的SSIM
    for _ in range(levels):
        #求每一对小窗口的结构相似性指数（SSIM）
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        # print("sim", sim)
        mssim.append(sim)
        mcs.append(cs)

        #以求最大池的方式移动图像img1, img2的位置
        img1 = F.avg_pool2d(img1, (2, 2)) #平均池化。 （2，2）：stride横向、纵向都步长为2.
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim) #torch.stack()保留序列、张量矩阵信息，将一个个张量按照时间序列排序，拼接成一个三维立体。   扩张维度。
    mcs = torch.stack(mcs)

    #避免当两张图像都有非常小的MS-SSIM时，无法继续训练。
    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2 #mssim+1: 将mmsim中的每个元素都加1.
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1]) #pow1的所有行列 * pow2改成一串。  返回输入tensor的所有原始的乘积
    return output


#Structural similarity index 结构相似性指标
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM  #assume：假定
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
@LOSS_REGISTRY.register()
#多尺度结构相似性
class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):  #size_average求出每个小窗口的相似性后，要计算所有窗口相似性的平均值，作为整个图像的相似性指标。
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average, normalize=True)

@LOSS_REGISTRY.register()
#Welsch
class WelschLoss(torch.nn.Module):
    def __init__(self, alpha=0.1, weight=1):  #size_average求出每个小窗口的相似性后，要计算所有窗口相似性的平均值，作为整个图像的相似性指标。
        super(WelschLoss, self).__init__()
        self.weight = weight
        self.alpha2 = alpha*alpha

    def forward(self, img1, img2):
        # TODO: store window between calls if possible,
        # return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)
        return self.alpha2*(1-torch.exp(-(img1-img2)^2/(self.alpha2*2)))

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, y, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        y_masked = torch.gather(y, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=x.device)
        # mask[:, :len_keep] = 0
        # # unshuffle to get the binary mask
        # mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, y_masked, ids_keep

    def forward_encoder(self, x, y, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        y = self.patch_embed(y)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        y = y + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, y, ids_keep = self.random_masking(x, y, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x_out = []
        max_numbers = 6
        for blk in self.blocks[:max_numbers]:
            x = blk(x)
            x_out.append(x)
        # x = self.norm(x)

        y = torch.cat((cls_tokens, y), dim=1)
        y_out = []
        for blk in self.blocks[:max_numbers]:
            y = blk(y)
            y_out.append(y)
        # y = self.norm(y)

        # return x_out[:6][:, :, :], y_out[:6][:, :, :], ids_keep
        return x_out, y_out, ids_keep

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, pimgs, gtimgs, mask_ratio=0.75):
        platent, glatent, ids = self.forward_encoder(pimgs, gtimgs, mask_ratio)

        return platent, glatent, ids

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

import math
import torch.nn as nn

def interpolate_pos_encoding(npatch, dim, model, patch_size, stride_hw, w: int, h: int) -> torch.Tensor:

    N = model.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return model.pos_embed
    class_pos_embed = model.pos_embed[:, 0]
    patch_pos_embed = model.pos_embed[:, 1:]
    # compute number of tokens taking stride into account
    w0 = 1 + (w - patch_size) // stride_hw[1]
    h0 = 1 + (h - patch_size) // stride_hw[0]
    assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                    stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode='bicubic',
        align_corners=False, recompute_scale_factor=False
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
class selfPerceptualLoss(nn.Module):
    def __init__(self, img_size):
        super(selfPerceptualLoss, self).__init__()
        self.l2_loss = MSELoss()
        self.l1_loss = L1Loss(reduction='mean')

    def forward(self, query, positive_key, ids):
        output = 0
        for q, p in zip(query, positive_key):
            output += self.l1_loss(q, p)
        return output

@LOSS_REGISTRY.register()
class ReconstructPerceptualLoss(nn.Module):
    def __init__(self, loss_weight):
        super(ReconstructPerceptualLoss, self).__init__()
        self.weight=loss_weight
        img_size = 256
        pretrained_ckpt = "mae_pretrain_vit_base.pth"
        self.pretrain_mae = mae_vit_base_patch16_dec512d8b(img_size=img_size)
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(self.pretrain_mae, checkpoint_model)
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False
        self.constrastive_loss = selfPerceptualLoss(img_size)
        # self.constrastive_loss = projectedDistributionLoss()

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward(self, recover_img, gt):


        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt, 0.50)
        # Local MAE
        contrast_loss = self.constrastive_loss(predict_embed, gt_embed, ids)

        # Global MAE
        # contrast_loss =0
        # for predict_e, gt_e in zip(predict_embed, gt_embed):
        #    contrast_loss += projectedDistributionLoss(predict_e, gt_e) * 1e-4


        return self.weight*contrast_loss