import torch
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
from enum import Enum


class ResNetType(Enum):
    UPSAMPLE = 0
    DOWNSAMPLE = 1
    SAME = 2


class ResnetBlock(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, block_type, use_bnorm=True, use_bias=True, layers_per_group=1, use_3D=False):
        super().__init__()
        assert isinstance(block_type, ResNetType)
        self.type = block_type

        self._use_bnorm = use_bnorm
        
        if use_3D:
            self._conv1 = nn.Conv3d(int(in_channels), int(out_channels), 3, bias=use_bias, padding=1)
            self._conv2 = nn.Conv3d(int(out_channels), int(out_channels), 3, bias=use_bias, padding=1)
        else:
            self._conv1 = nn.Conv2d(int(in_channels), int(out_channels), 3, bias=use_bias, padding=1)
            self._conv2 = nn.Conv2d(int(out_channels), int(out_channels), 3, bias=use_bias, padding=1)

        if self._use_bnorm:
            self._bn1 = nn.GroupNorm(in_channels//layers_per_group, in_channels)#nn.BatchNorm2d(in_channels)
            self._bn2 = nn.GroupNorm(out_channels//layers_per_group, out_channels)#nn.BatchNorm2d(out_channels)

        self._relu1 = nn.LeakyReLU(0.2)
        self._relu2 = nn.LeakyReLU(0.2)
        self._resample = None

        if block_type == ResNetType.UPSAMPLE:
            if use_3D:
                self._resample = nn.Upsample(scale_factor=2, mode='trilinear')
            else:
                self._resample = nn.UpsamplingBilinear2d(scale_factor=2)

        elif block_type == ResNetType.DOWNSAMPLE:
            if use_3D:
                self._resample = nn.AvgPool3d(2)
                self._resample2 = nn.AvgPool3d(2)
            else:
                self._resample = nn.AvgPool2d(2)
                self._resample2 = nn.AvgPool2d(2)

        if use_3D:
            self._resid_conv = nn.Conv3d(int(in_channels), int(out_channels), 1, bias=use_bias)
        else:
            self._resid_conv = nn.Conv2d(int(in_channels), int(out_channels), 1, bias=use_bias)

    def forward(self, input):
        resid_connection = None

        if self.type == ResNetType.UPSAMPLE or self.type == ResNetType.SAME:
            if self._resample:
                input = self._resample(input)
            resid_connection = self._resid_conv(input)

        net = input
        if self._use_bnorm:
            net = self._bn1(net)
        net = self._relu1(net)
        net = self._conv1(net)

        if self._use_bnorm:
            net = self._bn2(net)
        net = self._relu2(net)
        net = self._conv2(net)

        if self.type == ResNetType.DOWNSAMPLE:
            net = self._resample(net)
            resid_connection = self._resid_conv(input)
            resid_connection = self._resample2(resid_connection)

        net = net + resid_connection
        return net


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_dim, activation=nn.SiLU()):
        super(SelfAttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1)+1e-5)
        self.activation = activation

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, use_group_norm=False, activation=nn.SiLU()):
        super(ConvBlock, self).__init__()
        self.use_group_norm = use_group_norm
        self.conv = nn.Conv2d(out_dim, out_dim, kernel_size, stride, padding)
        self.conv01 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv02 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.activation = activation

        if self.use_group_norm:
            self.gn = nn.GroupNorm(ceil(in_dim/8), out_dim) #nn.BatchNorm2d(out_dim)
            self.gn01 = nn.GroupNorm(ceil(in_dim/8), out_dim) #nn.BatchNorm2d(out_dim)
            self.gn02 = nn.GroupNorm(ceil(in_dim/8), out_dim) #nn.BatchNorm2d(out_dim)
            self.gn1  = nn.GroupNorm(ceil(in_dim/8), out_dim) #nn.BatchNorm2d(out_dim)

    def forward(self,x):
        x0 = x
        x = self.activation(self.conv01(x))
        if self.use_group_norm:
            x = self.gn(x)
        x1 = x
        x = self.activation(self.conv02(x))
        if self.use_group_norm:
            x = self.gn01(x)
        if x0.shape[1]==1:
            x = self.activation(self.conv(x+x0+x1))
        else:
            x = x1 + x
            x[:,:x0.shape[1],:,:] = x0
            x = self.activation(self.conv(x))
        if self.use_group_norm:
            #x = self.gn02(x)
            return self.gn1(x)
            #return self.gn1(nn.AvgPool2d(2)(x0 + x))
        else:
            return x
            #return x0 + x


class DeconvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, output_padding=1, use_group_norm=True,end=False,var=False, activation=nn.SiLU()):
        super(DeconvBlock, self).__init__()
        self.use_group_norm = use_group_norm
        self.end = end
        self.var = var
        self.activation = activation
        if self.end:
            self.gamma = nn.Parameter(1e-5*torch.ones(1))

        if end:
            self.deconv = nn.ConvTranspose2d(in_dim, int(in_dim/2), kernel_size, stride, padding,output_padding)
            self.deconv01 = nn.Conv2d(int(in_dim/2), out_dim, 3, 1, 1)
            self.deconv02 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            self.deconv03 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            self.deconv03.weight.data = self.deconv03.weight.data*1e-3
            self.deconv03.bias.data = self.deconv03.bias.data*1e-3
        else:
            self.deconv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding,output_padding)
            self.deconv01 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            self.deconv02 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        if self.use_group_norm:
            if self.end:
                self.gn = nn.GroupNorm(ceil(in_dim/8), int(in_dim/2))
                in_dim = 1
            else:
                self.gn = nn.GroupNorm(ceil(in_dim/8), out_dim)
            self.gn01 = nn.GroupNorm(ceil(in_dim/8), out_dim)
            self.gn02 = nn.GroupNorm(ceil(in_dim/8), out_dim)
            self.gn1  = nn.GroupNorm(ceil(in_dim/8), out_dim)


    def forward(self,x):
        x = self.activation(self.deconv(x))
        if self.use_group_norm:
            x = self.gn(x)
        x0 = x
        x = self.activation(self.deconv01(x))
        if self.use_group_norm:
            x = self.gn01(x)
        x = self.activation(self.deconv02(x))
        if self.use_group_norm:
            x = self.gn02(x)
        if self.end:
            x = self.gamma*self.deconv03(x)
            if self.var:
                return x
            else:
                return x 
        if self.use_group_norm:
            return self.gn1(x0 + x)
        else:
            return x0 + x