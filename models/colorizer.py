import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import one_hot
from .deform_im2col_util import deform_im2col
import pdb


from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

import spatial_correlation_sampler_backend as correlation
def one_hot(labels, C):
    one_hot = torch.zeros(labels.size(0), C, labels.size(2), labels.size(3))
    if labels.is_cuda: one_hot = one_hot.cuda()

    target = one_hot.scatter_(1, labels, 1)
    if labels.is_cuda: target = target.cuda()

    return target

def spatial_correlation_sample(input1,
                               input2,
                               kernel_size=1,
                               patch_size=1,
                               stride=1,
                               padding=0,
                               dilation=1,
                               dilation_patch=1):
    """Apply spatial correlation sampling on from input1 to input2,
    Every parameter except input1 and input2 can be either single int
    or a pair of int. For more information about Spatial Correlation
    Sampling, see this page.
    https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/
    Args:
        input1 : The first parameter.
        input2 : The second parameter.
        kernel_size : total size of your correlation kernel, in pixels
        patch_size : total size of your patch, determining how many
            different shifts will be applied
        stride : stride of the spatial sampler, will modify output
            height and width
        padding : padding applied to input1 and input2 before applying
            the correlation sampling, will modify output height and width
        dilation_patch : step for every shift in patch
    Returns:
        Tensor: Result of correlation sampling
    """
    return SpatialCorrelationSamplerFunction.apply(input1, input2,
                                                   kernel_size, patch_size,
                                                   stride, padding, dilation, dilation_patch)


class SpatialCorrelationSamplerFunction(Function):

    @staticmethod
    def forward(ctx,
                input1,
                input2,
                kernel_size=1,
                patch_size=1,
                stride=1,
                padding=0,
                dilation=1,
                dilation_patch=1):

        ctx.save_for_backward(input1, input2)
        kH, kW = ctx.kernel_size = _pair(kernel_size)
        patchH, patchW = ctx.patch_size = _pair(patch_size)
        padH, padW = ctx.padding = _pair(padding)
        dilationH, dilationW = ctx.dilation = _pair(dilation)
        dilation_patchH, dilation_patchW = ctx.dilation_patch = _pair(dilation_patch)
        dH, dW = ctx.stride = _pair(stride)

        output = correlation.forward(input1, input2,
                                     kH, kW, patchH, patchW,
                                     padH, padW, dilationH, dilationW,
                                     dilation_patchH, dilation_patchW,
                                     dH, dW)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_variables

        kH, kW = ctx.kernel_size
        patchH, patchW = ctx.patch_size
        padH, padW = ctx.padding
        dilationH, dilationW = ctx.dilation
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride

        grad_input1, grad_input2 = correlation.backward(input1, input2, grad_output,
                                                        kH, kW, patchH, patchW,
                                                        padH, padW, dilationH, dilationW,
                                                        dilation_patchH, dilation_patchW,
                                                        dH, dW)
        return grad_input1, grad_input2, None, None, None, None, None, None


class SpatialCorrelationSampler(nn.Module):
    def __init__(self, kernel_size=1, patch_size=1, stride=1, padding=0, dilation=1, dilation_patch=1):
        super(SpatialCorrelationSampler, self).__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch

    def forward(self, input1, input2):
        return SpatialCorrelationSamplerFunction.apply(input1, input2, self.kernel_size,
                                                       self.patch_size, self.stride,
                                                       self.padding, self.dilation, self.dilation_patch)

class Colorizer(nn.Module):
    def __init__(self, D=4, R=6, C=32):
        super(Colorizer, self).__init__()
        self.D = D
        self.R = R  # window size
        self.C = C

        self.P = self.R * 2 + 1
        self.N = self.P * self.P
        self.count = 0

        self.memory_patch_R = 12
        self.memory_patch_P = self.memory_patch_R * 2 + 1
        self.memory_patch_N = self.memory_patch_P * self.memory_patch_P

        self.correlation_sampler_dilated = [
            SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.memory_patch_P,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=dirate) for dirate in range(2,6)]

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.P,
            stride=1,
            padding=0,
            dilation=1)

    def prep(self, image, HW):
        #print('*********************8',image.size())
        _,c,_,_ = image.size()

        x = image.float()[:,:,::self.D,::self.D]

        if c == 1 :
            x = one_hot(x.long(), self.C)

        return x

    def forward(self, feats_r, feats_t, quantized_r, ref_index, current_ind, dil_int = 15):
        """
        Warp y_t to y_(t+n). Using similarity computed with im (t..t+n)
        :param feats_r: f([im1, im2, im3])
        :param quantized_r: [y1, y2, y3]
        :param feats_t: f(im4)
        :param mode:
        :return:
        """
        # For frame interval < dil_int, no need for deformable resampling
        nref = len(feats_r)
        nsearch = len([x for x in ref_index if current_ind - x > dil_int])

        # The maximum dilation rate is 4
        dirates = [ min(4, (current_ind - x) // dil_int +1) for x in ref_index if current_ind - x > dil_int]
        b,c,h,w = feats_t.size()
        N = self.P * self.P
        corrs = []

        # offset0 = []
        for searching_index in range(nsearch):
            ##### GET OFFSET HERE.  (b,h,w,2)
            samplerindex = dirates[searching_index]-2
            coarse_search_correlation = self.correlation_sampler_dilated[samplerindex](feats_t, feats_r[searching_index])  # b, p, p, h, w
            coarse_search_correlation = coarse_search_correlation.reshape(b, self.memory_patch_N, h*w)
            coarse_search_correlation = F.softmax(coarse_search_correlation, dim=1)
            coarse_search_correlation = coarse_search_correlation.reshape(b,self.memory_patch_P,self.memory_patch_P,h,w,1)
            _y, _x = torch.meshgrid(torch.arange(-self.memory_patch_R,self.memory_patch_R+1),torch.arange(-self.memory_patch_R,self.memory_patch_R+1))
            grid = torch.stack([_x, _y], dim=-1).unsqueeze(-2).unsqueeze(-2)\
                .reshape(1,self.memory_patch_P,self.memory_patch_P,1,1,2).contiguous().float().to(coarse_search_correlation.device)
            offset0 = (coarse_search_correlation * grid ).sum(1).sum(1) * dirates[searching_index]  # 1,h,w,2

            col_0 = deform_im2col(feats_r[searching_index], offset0, kernel_size=self.P)  # b,c*N,h*w
            col_0 = col_0.reshape(b,c,N,h,w)
            ##
            corr = (feats_t.unsqueeze(2) * col_0).sum(1)   # (b, N, h, w)

            corr = corr.reshape([b, self.P * self.P, h * w])
            corrs.append(corr)

        for ind in range(nsearch, nref):
            corrs.append(self.correlation_sampler(feats_t, feats_r[ind]))
            _, _, _, h1, w1 = corrs[-1].size()
            corrs[ind] = corrs[ind].reshape([b, self.P*self.P, h1*w1])

        corr = torch.cat(corrs, 1)  # b,nref*N,HW
        corr = F.softmax(corr, dim=1)
        corr = corr.unsqueeze(1)
        #print(len(quantized_r),quantized_r[0].shape)
        #print(h,w)
        #print('~~~~~~~~~~~',self.training)
        qr = [self.prep(qr, (h,w)) for qr in quantized_r]
        #print(h,w)
        #print(len(quantized_r),quantized_r[0].shape)
        #print('******************',len(qr),qr[0].shape)
        im_col0 = [deform_im2col(qr[i], offset0, kernel_size=self.P)  for i in range(nsearch)]# b,3*N,h*w
        im_col1 = [F.unfold(r, kernel_size=self.P, padding =self.R) for r in qr[nsearch:]]
        image_uf = im_col0 + im_col1

        image_uf = [uf.reshape([b,qr[0].size(1),self.P*self.P,h*w]) for uf in image_uf]
        image_uf = torch.cat(image_uf, 2)
        out = (corr * image_uf).sum(2).reshape([b,qr[0].size(1),h,w])

        return out

def torch_unravel_index(indices, shape):
    rows = indices / shape[0]
    cols = indices % shape[1]

    return (rows, cols)
