from torch import nn as nn
from torch.nn import functional as F
import torch
from basicsr.utils.registry import LOSS_REGISTRY
from torch.autograd import Variable
from math import exp
import lpips

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



