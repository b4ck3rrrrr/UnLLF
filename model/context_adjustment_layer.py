import torch
from torch import nn, Tensor
from torch.nn.utils import weight_norm

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from torchvision.ops import deform_conv2d

import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class SobelConv(nn.Module):
    """不可学习 Sobel 边缘卷积"""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kx', kernel_x)
        self.register_buffer('ky', kernel_y)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        # x: [N,3,H,W]
        gx = F.conv2d(self.pad(x[:, :1]), self.kx, padding=0)
        gy = F.conv2d(self.pad(x[:, :1]), self.ky, padding=0)
        return torch.cat([gx, gy], dim=1)   # [N,2,H,W]


class LiteDCN(nn.Module):
    """仅预测 3×3 偏移，不预测 mask，省计算"""
    def __init__(self, c_in, c_out, kernel=3, stride=1, padding=1):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.kernel = kernel
        self.stride, self.padding = stride, padding
        # 偏移场：2*K*K 个数
        self.offset_conv = nn.Conv2d(c_in, 2 * kernel * kernel, 3, padding=1, bias=False)
        # 实际权重
        self.weight = nn.Parameter(torch.randn(c_out, c_in, kernel, kernel) * 0.01)
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x, offset=None):
        # 如果外部没给 offset，就自己算
        if offset is None:
            offset = self.offset_conv(x)
        out = deform_conv2d(x, offset, self.weight, self.bias,
                            stride=self.stride, padding=self.padding)
        return out


# ---------------- 像素-自适应重组 (PAR) ----------------
class PAR(nn.Module):
    """用 edge 生成偏移，对 feat 做可变形重组"""
    def __init__(self, c):
        super().__init__()
        self.dcn = LiteDCN(c, c, 3, 1, 1)
        self.edge_conv = nn.Sequential(
            nn.Conv2d(2, c, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 2 * 3 * 3, 3, 1, 1)  # 仅输出偏移
        )

    def forward(self, feat, edge):
        offset = self.edge_conv(edge)         # 用 edge 引导偏移
        return self.dcn(feat, offset)


# ---------------- 区域-置信门控 (RCG) ----------------
class RCG(nn.Module):
    """大感受野融合 + 动态门控"""
    def __init__(self, c):
        super().__init__()
        self.gather = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=d, dilation=d, bias=False) for d in [1, 2, 4]
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 3, c, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        xs = [m(x) for m in self.gather]
        xs = torch.cat(xs, dim=1)
        xs = self.fuse(xs)
        g = self.gate(x)
        return x + g * xs


# ---------------- Split-GRU ----------------
class SplitGRUCell(nn.Module):
    """通道劈半，只更新一半，轻量且抗梯度消失"""
    def __init__(self, c):
        super().__init__()
        half = c // 2
        self.convz = nn.Conv2d(c * 2, half, 3, 1, 1, bias=False)
        self.convr = nn.Conv2d(c * 2, half, 3, 1, 1, bias=False)
        self.convq = nn.Conv2d(c * 2, half, 3, 1, 1, bias=False)

    def forward(self, x, h):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        # 把 r 从 [B, half, H, W] 扩展到 [B, c, H, W]
        r = r.repeat_interleave(h.shape[1] // r.shape[1], dim=1)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        half = h.shape[1] // 2
        h1, h2 = h.split([half, h.shape[1] - half], dim=1)
        return torch.cat([(1 - z) * h1 + z * q, h2], dim=1)


# ---------------- 创新细化模块 ----------------
class ContextAdjustmentLayer(nn.Module):
    def __init__(self,
                 num_blocks=6,
                 feature_dim=16,
                 use_edge=True):
        super().__init__()
        self.use_edge = use_edge

        # 边缘提取
        self.sobel = SobelConv() if use_edge else None

        # 输入映射
        in_ch = 4 + (2 if use_edge else 0)
        self.in_conv = nn.Conv2d(in_ch, feature_dim, 3, 1, 1)

        # PAR + RCG 交替堆叠
        self.stages = nn.ModuleList()
        for _ in range(num_blocks):
            self.stages.append(nn.ModuleDict({
                'par': PAR(feature_dim),
                'rcg': RCG(feature_dim)
            }))

        # Split-GRU
        self.gru = SplitGRUCell(feature_dim)
        self.h = None

        # 残差-增量头
        self.delta_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, 3, 1, 1),
            nn.Tanh()
        )

        # 动态缩放 λ：根据局部方差生成
        self.lambda_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid()
        )

        # 占用头（同原，加边缘权重）
        self.occ_head = nn.Sequential(
            weight_norm(nn.Conv2d(1 + 3, feature_dim, 3, 1, 1)),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)), nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)),
            weight_norm(nn.Conv2d(feature_dim, 1, 3, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, disp_raw, occ_raw, img):
        B, _, H, W = disp_raw.shape

        # 1. 边缘先验
        edge = self.sobel(img) if self.use_edge else None
        x = torch.cat([disp_raw, img, edge], 1) if self.use_edge else torch.cat([disp_raw, img], 1)
        feat = self.in_conv(x)

        # 2. 初始化隐藏状态
        if self.h is None or self.h.shape[-2:] != (H, W):
            self.h = torch.zeros_like(feat)
        self.h = self.gru(feat, self.h)

        feat = self.h

        # 3. PAR-RCG 交替
        for stage in self.stages:
            feat = stage['par'](feat, edge)   # 用 edge 重组
            feat = stage['rcg'](feat)         # 大感受野门控

        # 4. 残差-增量
        delta = self.delta_conv(feat)
        lam = self.lambda_conv(disp_raw)      # 根据 disp 局部方差动态缩放
        disp_final = disp_raw + lam * delta

        # 5. 占用分支（加边缘权重）
        occ_edge_weight = 1 + torch.sigmoid(edge[:, :1] + edge[:, 1:]) if self.use_edge else 1
        occ_feat = torch.cat([occ_raw, img], 1)
        occ_final = self.occ_head(occ_feat) * occ_edge_weight
        self.h = self.h.detach()
        return disp_final, occ_final













# ------------------ 基础单元 ------------------
# class SobelConv(nn.Module):
#     """不可学习 Sobel 边缘卷积"""
#     def __init__(self):
#         super().__init__()
#         kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         self.register_buffer('kx', kernel_x)
#         self.register_buffer('ky', kernel_y)
#         self.pad = nn.ReflectionPad2d(1)

#     def forward(self, x):
#         # x: [N,3,H,W]
#         gx = F.conv2d(self.pad(x[:, :1]), self.kx, padding=0)
#         gy = F.conv2d(self.pad(x[:, :1]), self.ky, padding=0)
#         return torch.cat([gx, gy], dim=1)  # [N,2,H,W]

# class MultiDilationResBlock(nn.Module):
#     """并联 4 路空洞卷积 + 1×1 融合，等效大感受野"""
#     def __init__(self, c, expansion=3):
#         super().__init__()
#         inner = int(c * expansion)
#         self.branch = nn.ModuleList([
#             nn.Conv2d(c, inner, 3, padding=d, dilation=d, bias=False) for d in [1, 2, 4, 8]
#         ])
#         self.fuse = nn.Sequential(
#             nn.Conv2d(inner * 4, c, 1, bias=False),
#             nn.ReLU(inplace=True)
#         )
#         self.gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c, c, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x, disp):
#         # disp 仅用于 residual shortcut，不参与计算
#         b = torch.cat([m(x) for m in self.branch], dim=1)
#         b = self.fuse(b)
#         g = self.gate(x)
#         return x + g * b  # 残差加门控

# class ConvGRUCell(nn.Module):
#     """轻量 2D Conv-GRU，仅 2 层卷积"""
#     def __init__(self, c):
#         super().__init__()
#         self.convz = nn.Conv2d(c * 2, c, 3, padding=1, bias=False)
#         self.convr = nn.Conv2d(c * 2, c, 3, padding=1, bias=False)
#         self.convq = nn.Conv2d(c * 2, c, 3, padding=1, bias=False)

#     def forward(self, x, h):
#         # x: input feature, h: hidden state
#         hx = torch.cat([h, x], dim=1)
#         z = torch.sigmoid(self.convz(hx))
#         r = torch.sigmoid(self.convr(hx))
#         q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
#         return (1 - z) * h + z * q

# # ------------------ 改进版细化层 ------------------
# class ContextAdjustmentLayer(nn.Module):
#     def __init__(self,
#                  num_blocks=8,
#                  feature_dim=16,
#                  expansion=3,
#                  use_edge=True,
#                  use_large_kernel=True,
#                  use_gru=True,
#                  use_gamma=True):
#         super().__init__()
#         self.use_edge = use_edge
#         self.use_large_kernel = use_large_kernel
#         self.use_gru = use_gru
#         self.use_gamma = use_gamma

#         # --- disp head ---
#         in_ch = 4 + (2 if use_edge else 0)
#         self.in_conv = nn.Conv2d(in_ch, feature_dim, 3, padding=1)
#         Block = MultiDilationResBlock if use_large_kernel else ResBlock  # 回退原版
#         self.layers = nn.ModuleList([Block(feature_dim, expansion) for _ in range(num_blocks)])
#         self.out_conv = nn.Conv2d(feature_dim, 1, 3, padding=1)
#         if use_gamma:
#             self.gamma = nn.Parameter(torch.zeros(1))  # 初始 0

#         # --- occ head ---
#         occ_in = 1 + 3
#         self.occ_head = nn.Sequential(
#             weight_norm(nn.Conv2d(occ_in, feature_dim, 3, padding=1)),
#             weight_norm(nn.Conv2d(feature_dim, feature_dim, 3, padding=1)),
#             nn.ReLU(inplace=True),
#             weight_norm(nn.Conv2d(feature_dim, feature_dim, 3, padding=1)),
#             weight_norm(nn.Conv2d(feature_dim, feature_dim, 3, padding=1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_dim, 1, 3, padding=1),
#             nn.Sigmoid()
#         )

#         # --- 边缘提取 ---
#         if use_edge:
#             self.sobel = SobelConv()

#         # --- Conv-GRU ---
#         if use_gru:
#             self.gru_h = None  # 状态缓存
#             self.gru_cell = ConvGRUCell(feature_dim)

#     def forward(self, disp_raw: Tensor, occ_raw: Tensor, img: Tensor):
#         """
#         参数同原版，新增 e_raw: [N,1,H,W] 为左右 warp 误差图，可不传
#         """
#         # 1. 边缘图
#         print('disp_raw',disp_raw.shape)
#         print('occ_raw',occ_raw.shape)
#         print('img',img.shape)
#         edge = self.sobel(img) if self.use_edge else None
#         x = torch.cat([disp_raw, img, edge], dim=1) if self.use_edge else torch.cat([disp_raw, img], dim=1)

#         # 2. 特征提取
#         feat = self.in_conv(x)

#         self.gru_h = torch.zeros_like(feat)
           
#         self.gru_h = self.gru_cell(feat, self.gru_h)
#         feat = self.gru_h

#         # 4. 残差主干
#         for layer in self.layers:
#             feat = layer(feat, disp_raw)

#         # 5. 残差输出 + 缩放
#         disp_res = self.out_conv(feat)
#         if self.use_gamma:
#             disp_res = self.gamma * disp_res
#         disp_final = disp_raw + disp_res

#         # 6. occ 分支
#         occ_final = self.occ_head(torch.cat([occ_raw, img], dim=1))

#         return disp_final, occ_final








# class ContextAdjustmentLayer(nn.Module):
#     """
#     Adjust the disp and occ based on image context, design loosely follows https://github.com/JiahuiYu/wdsr_ntire2018
#     """

#     def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
#         super().__init__()
#         self.num_blocks = num_blocks

#         # disp head
#         self.in_conv = nn.Conv2d(4, feature_dim, kernel_size=3, padding=1)
#         self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
#         self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)

#         # occ head
#         self.occ_head = nn.Sequential(
#             weight_norm(nn.Conv2d(1 + 3, feature_dim, kernel_size=3, padding=1)),
#             weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
#             nn.ReLU(inplace=True),
#             weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
#             weight_norm(nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, disp_raw: Tensor, occ_raw: Tensor, img: Tensor):
#         """
#         :param disp_raw: raw disparity, [N,1,H,W]
#         :param occ_raw: raw occlusion mask, [N,1,H,W]
#         :param img: input left image, [N,3,H,W]
#         :return:
#             disp_final: final disparity [N,1,H,W]
#             occ_final: final occlusion [N,1,H,W] 
#         """""
#         feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
#         for layer in self.layers:
#             feat = layer(feat, disp_raw)
#         disp_res = self.out_conv(feat)
#         disp_final = disp_raw + disp_res

#         occ_final = self.occ_head(torch.cat([occ_raw, img], dim=1))

#         return disp_final, occ_final

class ContextAdjustmentLayerv2(nn.Module):
    """
    Adjust the disp and occ based on image context, design loosely follows https://github.com/JiahuiYu/wdsr_ntire2018
    """

    def __init__(self, num_blocks=8, feature_dim=16, expansion=3):
        super().__init__()
        self.num_blocks = num_blocks

        # disp head
        self.in_conv = nn.Conv2d(4, feature_dim, kernel_size=3, padding=1)
        self.layers = nn.ModuleList([ResBlock(feature_dim, expansion) for _ in range(num_blocks)])
        self.out_conv = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)


    def forward(self, disp_raw: Tensor,  img: Tensor):
        """
        :param disp_raw: raw disparity, [N,1,H,W]
        :param occ_raw: raw occlusion mask, [N,1,H,W]
        :param img: input left image, [N,3,H,W]
        :return:
            disp_final: final disparity [N,1,H,W]
            occ_final: final occlusion [N,1,H,W] 
        """""
        feat = self.in_conv(torch.cat([disp_raw, img], dim=1))
        for layer in self.layers:
            feat = layer(feat, disp_raw)
        disp_res = self.out_conv(feat)
        disp_final = disp_raw + disp_res
 

        return disp_final

class ResBlock(nn.Module):
    def __init__(self, n_feats: int, expansion_ratio: int, res_scale: int = 1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats + 1, n_feats * expansion_ratio, kernel_size=3, padding=1)),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1))
        )

    def forward(self, x: torch.Tensor, disp: torch.Tensor):
        return x + self.module(torch.cat([disp, x], dim=1)) * self.res_scale

