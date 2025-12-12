import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import math
import timm
import torch.nn.functional as F

from pytorch_wavelets import DWTForward
from INDWT import DWTInverse_new



def get_convnext(model_name='convnext_base', pretrained=True, num_classes=2):
    """
    :param model_name: convnext_base
    :param pretrained:
    :param num_classes:
    :return:
    """
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    return net







class DCT(nn.Module):
    def __init__(self, N = 8, in_channal = 3):
        super(DCT, self).__init__()
        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N  #8*8=64
        self.in_channal = in_channal  #3
        self.out_channal =  N * N * in_channal #8*8*3=192

        # 3 H W -> N*N*in_channel  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)
        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.mk_coff(N = N)).float().unsqueeze(1)
        self.dct_conv.weight.data = torch.cat([self.weight]*self.in_channal, dim=0) # 64 1 8 8
        self.dct_conv.weight.requires_grad = False


        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])  # a simillar version, maybe be a little wrong
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix

    def forward(self, x):
        '''

        :param x: B C H W, 0-1. RGB  YCbCr:  b c h w, YCBCR  DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H
        :return:
        '''
        # jpg = (jpg * self.std) + self.mean # 0-1
        ycbcr = self.Ycbcr(x)  # b 3 h w
        dct = self.dct_conv(ycbcr)
        return ycbcr,dct

    def reverse(self, x):
        ycbcr = F.conv_transpose2d(x, torch.cat([self.weight] * 3, 0),
                                   bias=None, stride=8, groups=3)
        rgb = self.reYcbcr(ycbcr)
        return rgb, ycbcr

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

class ReDCT(nn.Module):
    def __init__(self, N = 8, in_channal = 3):
        super(ReDCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.in_channal = in_channal * N * N
        self.out_channal = in_channal
        self.fre_len = N * N

        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)


        self.reDCT = nn.ConvTranspose2d(self.in_channal, self.out_channal, self.N,  self.N, bias = False, groups=self.out_channal)
        self.reDCT.weight.data = torch.cat([self.weight]*self.out_channal, dim=0)
        self.reDCT.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix


    def forward(self, dct):
        '''
        IDCT  from DCT domain to pixle domain
        B C*64 H//8 W//8   ->   B C H W
        '''
        ycbcr = self.reDCT(dct)
        out=self.reYcbcr(ycbcr)

        return out,ycbcr

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth
    

### --- channel attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)



###### PAM_fre
class PAM_fre(nn.Module):

    def __init__(self, in_dim):

        super(PAM_fre, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_dim*2,
            out_channels= 2,
            kernel_size=3,
            padding=1
        )

        self.v_rgb = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)
        self.v_freq = nn.Parameter(torch.randn((1,in_dim,1,1)),requires_grad=True)

    def forward(self, rgb, freq):

        attmap = self.conv( torch.cat( (rgb,freq),1) )
        attmap = torch.sigmoid(attmap)

        rgb = attmap[:,0:1,:,:] * rgb * self.v_rgb + rgb  # rgb的权重应用到了RGB这个分支的
        freq = attmap[:,1:,:,:] * freq * self.v_freq + freq
        out = rgb + freq

        return out




#### ---位置注意力

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h       
        return out





#### --- 主函数， Convnext
class Conv_Net(nn.Module):
    def __init__(self, J=1, num_classes=2):
        super().__init__()
        self.name = 'convnext_base'
        self.convnext_rgb = get_convnext(model_name=self.name, num_classes=num_classes, pretrained=True)

        self.dct = DCT()
        self.idct = ReDCT()


        self.xfm1 = DWTForward(J=J, mode='zero', wave='haar')
        self.ifm1 = DWTInverse_new(mode='zero', wave='haar')

        channels = 192
        self.ca1 = ChannelAttention(channels)

        self.stem_dct_0 = self.convnext_rgb.stem
        self.dct_s0 = self.convnext_rgb.stages[0]
        self.stem_dwt_0 = self.convnext_rgb.stem
        self.dwt_s0 = self.convnext_rgb.stages[0]


        self.self_att = CoordAtt(inp=128, oup=128)  # 位置注意力

        self.pam_dct = PAM_fre(in_dim=128)   # 两个特征进行了cat
        self.pam_dwt = PAM_fre(in_dim=128)   # 两个特征进行了cat

        self.fusion=nn.Sequential(
            nn.Conv2d(128*2, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 112
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )




    def features(self, x):
        
        ycbcr, dctx = self.dct(x)
        ca = self.ca1(dctx)
        dct_choose = dctx * ca
        dctx_0, ycbcr2 = self.idct(dct_choose)   # dctx_0： 3 224 224


        Yl_1, Yh_1 = self.xfm1(x)
        Yl_11 = torch.zeros_like(Yl_1)   # 设置0矩阵
        hh =  Yh_1
        dwtx_0 = self.ifm1(Yl_11, hh)  # 恢复原始图像大小

        x = self.convnext_rgb.stem(x)
        dct_x0 = self.stem_dct_0(dctx_0)
        dwt_x0 = self.stem_dwt_0(dwtx_0)


        x = self.convnext_rgb.stages[0](x) # 128 56 56
        dct_x0 = self.dct_s0(dct_x0)
        dwt_x0 = self.dwt_s0(dwt_x0)

        x_self = self.self_att(x)  # 对RGB特征进行位置 att增强
        x_dct = self.pam_dct(x, dct_x0)   # rgb和fre特征进行交互
        x_dwt = self.pam_dwt(x, dwt_x0)
        f3 = torch.cat((x_dct, x_dwt),dim=1)

        x_fre = self.fusion(f3)
        

        x = x_fre + x_self + x
        

        x = self.convnext_rgb.stages[1](x)  # 256 28 28
        x = self.convnext_rgb.stages[2](x)
        x = self.convnext_rgb.stages[3](x)
  
        x = self.convnext_rgb.norm_pre(x)
        x = self.convnext_rgb.head(x)

        return x

    def classifier(self, fea):
        out, fea = self.xception_rgb.classifier(fea)
        return out, fea

    def forward(self, x):
        '''
        x: original rgb
        
        Return:
        out: (B, 2) the output for loss computing
        fea: (B, 1024) the flattened features before the last FC
        att_map: srm spatial attention map
        '''
        x = self.features(x)


        return x
    


if __name__ == '__main__':
     
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dummy = torch.rand((2, 3, 224, 224)).to(device)

    model_two = Conv_Net(num_classes=2).to(device)

    vit_out = model_two(dummy)  

    print(vit_out)
