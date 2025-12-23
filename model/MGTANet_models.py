import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from model.pvt_v2 import pvt_v2_b5
from torch.nn.parameter import Parameter
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from timm.models import create_model
from mmcv.cnn import build_norm_layer


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class _ASPP_attention(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(_ASPP_attention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(out_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * 2, out_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(out_dim),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim + out_dim * 3, out_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(out_dim),
            nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(3 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )
        self.cam4 = NonLocal_HW(out_dim)
        self.cam3 = NonLocal_HW(out_dim)
        self.cam2 = NonLocal_HW(out_dim)
        self.cam1 = NonLocal_HW(out_dim)
        self.convcat = nn.Sequential(
            nn.Conv2d(4 * out_dim, 2 * out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(2 * out_dim), nn.PReLU(),
            nn.Conv2d(2 * out_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1), nn.BatchNorm2d(out_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), 1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), 1))

        fusion4_123 = self.fuse4(torch.cat((conv1, conv2, conv3), 1))
        fusion3_124 = self.fuse3(torch.cat((conv1, conv2, conv4), 1))
        fusion2_134 = self.fuse2(torch.cat((conv1, conv3, conv4), 1))
        fusion1_234 = self.fuse1(torch.cat((conv2, conv3, conv4), 1))

        refine1 = self.cam1(fusion4_123)
        refine2 = self.cam2(fusion3_124)
        refine3 = self.cam3(fusion2_134)
        refine4 = self.cam4(fusion1_234)

        refine_fusion = torch.cat((refine1, refine2, refine3, refine4), 1)
        out = self.convcat(refine_fusion)
        return out


class NonLocal_HW(nn.Module):
    def __init__(self, channel):
        super(NonLocal_HW, self).__init__()
        self.inter_channel = channel // 2  # 通道缩减因子
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # 调整 conv_phi, conv_theta 和 conv_g 的形状
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)  # (b, inter_channel, H*W)
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2,
                                                                             1).contiguous()  # (b, H*W, inter_channel)
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  # (b, H*W, inter_channel)

        # 计算注意力矩阵
        mul_theta_phi = torch.matmul(x_theta, x_phi)  # (b, H*W, H*W)
        mul_theta_phi = self.softmax(mul_theta_phi)  # 对最后一个维度应用 softmax

        # 计算输出特征
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)  # (b, H*W, inter_channel)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h,
                                                                             w)  # (b, inter_channel, H, W)
        mask = self.conv_mask(mul_theta_phi_g)  # (b, c, H, W)

        out = mask + x  # 结合输入和输出特征
        return out


class NonLocal_C(nn.Module):
    def __init__(self, channel):
        super(NonLocal_C, self).__init__()
        self.inter_channel = channel // 2  # 通道缩减因子
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # 调整 conv_phi, conv_theta 和 conv_g 的形状
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)  # (b, inter_channel, H*W)
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  # (b, H*W, inter_channel)
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  # (b, H*W, inter_channel)

        # 计算注意力矩阵
        mul_theta_phi = torch.matmul(x_phi, x_theta)  # (b, inter_channel, inter_channel)
        mul_theta_phi = self.softmax(mul_theta_phi)  # 对最后一个维度应用 softmax

        # 计算输出特征
        mul_theta_phi_g = torch.matmul(x_g, mul_theta_phi)  # (b, H*W, inter_channel)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h,
                                                                             w)  # (b, inter_channel, H, W)
        mask = self.conv_mask(mul_theta_phi_g)  # (b, c, H, W)

        out = mask + x  # 结合输入和输出特征
        return out


class stage(nn.Module):
    def __init__(self, channel):
        super(stage, self).__init__()
        self.conv3 = BasicConv2d(channel, channel // 2, 3, 1, 1)
        self.conv1 = BasicConv2d(channel, channel // 2, 1, 1, 0)
        self.GAP_Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel, channel, 1, stride=1, bias=False),
            nn.Sigmoid()
        )
        self.lastconv = conv1 = BasicConv2d(channel, channel, 1, 1, 0)

    def forward(self, x):
        x3 = self.conv3(x)
        x1 = self.conv1(x)
        x_out = torch.cat((x3, x1), dim=1)
        y = self.GAP_Conv(x)
        out = x_out * y
        out = self.lastconv(out)
        return out


class DLDblock(nn.Module):
    def __init__(self, channel):
        super(DLDblock, self).__init__()
        self.preconv4 = BasicConv2d(512, channel, 3, 1, 1)
        self.preconv3 = BasicConv2d(320, channel, 3, 1, 1)
        self.preconv2 = BasicConv2d(128, channel, 3, 1, 1)
        self.preconv1 = BasicConv2d(64, channel, 3, 1, 1)

        self.stage4 = stage(channel)
        self.fuse4 = BasicConv2d(channel, channel, 1, 1, 0)

        self.stage3 = stage(channel)
        self.fuse3 = BasicConv2d(channel, channel, 1, 1, 0)

        self.stage2 = stage(channel)
        self.fuse2 = BasicConv2d(channel, channel, 1, 1, 0)

        self.stage1 = stage(channel)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.sal_4 = nn.Conv2d(64, 1, 1)
        self.sal_3 = nn.Conv2d(64, 1, 1)
        self.sal_2 = nn.Conv2d(64, 1, 1)
        self.sal_1 = nn.Conv2d(64, 1, 1)

    def forward(self, x4, x3, x2, x1):
        x4 = self.preconv4(x4)
        x3 = self.preconv3(x3)
        x2 = self.preconv2(x2)
        x1 = self.preconv1(x1)

        x4 = self.stage4(x4)
        x4_up1 = self.up2(x4)
        x4_cat = x4_up1 + x3

        x4_up2 = self.up4(x4)
        x4_up3 = self.up8(x4)

        x3 = self.stage3(x4_cat)
        x3_up1 = self.up2(x3)
        x3_cat = x3_up1 + x2 + x4_up2

        x3_up2 = self.up4(x3)

        x2 = self.stage2(x3_cat)
        x2 = self.up2(x2)
        x2_cat = x2 + x1 + x4_up3 + x3_up2

        x1 = self.stage1(x2_cat)
        x1 = self.up4(x1)

        out1 = self.sal_1(x1)
        out2 = self.sal_2(x2_cat)
        out3 = self.sal_3(x3_cat)
        out4 = self.sal_4(x4_cat)

        return out1, out2, out3, out4


class MCCNet_VGG(nn.Module):
    def __init__(self, channel=32):
        super(MCCNet_VGG, self).__init__()
        self.backbone = pvt_v2_b5()
        self.backbone.load_state_dict(torch.load('./model/pvt_v2_b5.pth'), strict=False)
        self.mf1 = _ASPP_attention(64, 64)
        self.mf2 = _ASPP_attention(128, 128)
        # self.mf3 = _ASPP_attention(320, 320)
        # self.mf4 = _ASPP_attention(512, 512)
        self.decoder = DLDblock(channel=64)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.nl3 = NonLocal_C(320)
        self.nl4 = NonLocal_C(512)

    def forward(self, x_rgb):
        F1, F2, F3, F4 = self.backbone(x_rgb)
        F1_mf = self.mf1(F1)
        F2_mf = self.mf2(F2)
        F3_nl = self.nl3(F3)
        F4_nl = self.nl4(F4)
        # F3_nl = self.nl3(F3)
        # F4_nl = self.nl4(F4)
        out1, out2, out3, out4 = self.decoder(F4_nl, F3_nl, F2_mf, F1_mf)
        return out1, self.up4(out2), self.up8(out3), self.up16(out4)
