"""
光计算加速的医学影像辅助诊断系统 —— 模型定义
================================================
基于 LTSimulator 光子计算模拟器平台
核心创新：用 nn.Linear 模拟卷积操作，所有核心计算均为光计算

模型架构：
  4层光模拟卷积 + 4层光池化 + 2层光全连接
  适配胸部 X 光片（128×128 输入，二分类：正常/肺炎）

光计算层：所有 nn.Linear 层
电计算层：仅 ReLU 激活函数（辅助非线性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpticalConv2d(nn.Module):
    """
    光模拟卷积模块
    原理：将输入图像划分为滑动窗口，每个窗口通过 Linear 映射到输出通道
    这相当于一个可学习的卷积核，但由光计算执行矩阵乘法
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        
        # 光计算权重：用 nn.Linear 模拟光域矩阵乘法
        self.optical_kernel = nn.Linear(
            in_channels * self.kernel_size[0] * self.kernel_size[1],
            out_channels,
            bias=True
        )
        
        # 预计算 unfold 参数
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        
        # 记录计算量（用于 Ro 计算）
        self.compute_amount = in_channels * self.kernel_size[0] * self.kernel_size[1] * out_channels
        
    def forward(self, x):
        batch, in_c, h, w = x.shape
        
        # 使用 Unfold 提取滑动窗口
        patches = self.unfold(x)  # [batch, in_c*k_h*k_w, L]
        
        # 内存布局优化
        patches = patches.transpose(1, 2).contiguous()  # [batch, L, in_c*k_h*k_w]
        
        # 光计算核心：矩阵乘法（在光域执行）
        output = self.optical_kernel(patches)  # [batch, L, out_c]
        
        # 输出重塑
        out_h = (h + 2*self.padding - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (w + 2*self.padding - self.kernel_size[1]) // self.stride[1] + 1
        
        output = output.transpose(1, 2).view(batch, self.out_channels, out_h, out_w)
        
        return output


class OpticalPool2d(nn.Module):
    """
    光计算池化模块
    创新点：池化操作也在光域完成
    原理：将池化窗口展平后通过 Linear 层映射为单输出
    """
    def __init__(self, channels, kernel_size=2):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        # 光计算池化：每个池化窗口的 kernel_size^2 个像素 → 1 个像素
        self.optical_pool = nn.Linear(
            self.kernel_size[0] * self.kernel_size[1], 
            1, 
            bias=False
        )
        
        # 记录计算量（用于 Ro 计算）
        self.compute_amount = channels * self.kernel_size[0] * self.kernel_size[1]
        
    def forward(self, x):
        batch, c, h, w = x.shape
        
        # 确保尺寸可以被 kernel_size 整除
        assert h % self.kernel_size[0] == 0, f"Height {h} not divisible by kernel_size {self.kernel_size[0]}"
        assert w % self.kernel_size[1] == 0, f"Width {w} not divisible by kernel_size {self.kernel_size[1]}"
        
        # 展开为池化窗口
        unfolded = x.unfold(2, self.kernel_size[0], self.kernel_size[0]).unfold(
            3, self.kernel_size[1], self.kernel_size[1]
        )
        # shape: [batch, c, out_h, out_w, k_h, k_w]
        
        unfolded = unfolded.contiguous().view(batch, c, -1, self.kernel_size[0] * self.kernel_size[1])
        # shape: [batch, c, L, k_h*k_w]
        
        # 光计算池化
        pooled = self.optical_pool(unfolded)  # [batch, c, L, 1]
        pooled = pooled.squeeze(-1)  # [batch, c, L]
        
        # 重塑输出
        out_h = h // self.kernel_size[0]
        out_w = w // self.kernel_size[1]
        pooled = pooled.view(batch, c, out_h, out_w)
        
        return pooled


class OpticalChestXRay(nn.Module):
    """
    光计算胸部 X 光肺炎诊断模型
    
    创新点：
    1. 4层光模拟卷积（逐步提取医学影像特征）
    2. 4层光池化（光域下采样，替代传统MaxPool/AvgPool）
    3. 2层全连接分类头（光域二分类）
    4. Dropout 正则化防止过拟合
    
    输入：1×128×128（灰度胸部X光片）
    输出：2（正常/肺炎）
    
    维度变化：
    Input:  [B, 1, 128, 128]
    Conv1:  [B, 16, 128, 128]   Pool1:  [B, 16, 64, 64]
    Conv2:  [B, 32, 64, 64]     Pool2:  [B, 32, 32, 32]
    Conv3:  [B, 64, 32, 32]     Pool3:  [B, 64, 16, 16]
    Conv4:  [B, 128, 16, 16]    Pool4:  [B, 128, 8, 8]
    Flatten → [B, 8192]
    FC0 → [B, 256] → Dropout → ReLU
    FC1 → [B, 2]
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        # ========== 光卷积-池化特征提取层 ==========
        # 第1层：1 → 16
        self.conv1 = OpticalConv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = OpticalPool2d(16, kernel_size=2)
        
        # 第2层：16 → 32
        self.conv2 = OpticalConv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = OpticalPool2d(32, kernel_size=2)
        
        # 第3层：32 → 64
        self.conv3 = OpticalConv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = OpticalPool2d(64, kernel_size=2)
        
        # 第4层：64 → 128
        self.conv4 = OpticalConv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = OpticalPool2d(128, kernel_size=2)
        
        # ========== 光全连接分类头 ==========
        # 经过4轮池化后：128 × 8 × 8 = 8192
        self.fc_input_dim = 128 * 8 * 8  # 8192
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 256),   # 光全连接层
            nn.Dropout(dropout_rate),              # 正则化（不作为光/电计算统计）
            nn.ReLU(),                            # 电计算层（仅辅助）
            nn.Linear(256, num_classes)           # 光全连接层
        )
        
        # 收集所有光计算层信息
        self._collect_optical_layers()
    
    def _collect_optical_layers(self):
        """收集所有光计算层（所有 nn.Linear 层）"""
        self.optical_layers = []
        
        # 卷积层中的 optical_kernel（4层）
        conv_layers = [
            ('conv1', self.conv1), ('conv2', self.conv2),
            ('conv3', self.conv3), ('conv4', self.conv4)
        ]
        for name, conv in conv_layers:
            self.optical_layers.append({
                "name": f"OpticalConv_{name}",
                "layer": conv.optical_kernel,
                "shape": f"({conv.optical_kernel.in_features}, {conv.optical_kernel.out_features})",
                "compute_amount": conv.compute_amount
            })
        
        # 池化层中的 optical_pool（4层）
        pool_layers = [
            ('pool1', self.pool1), ('pool2', self.pool2),
            ('pool3', self.pool3), ('pool4', self.pool4)
        ]
        for name, pool in pool_layers:
            self.optical_layers.append({
                "name": f"OpticalPool_{name}",
                "layer": pool.optical_pool,
                "shape": f"({pool.optical_pool.in_features}, {pool.optical_pool.out_features})",
                "compute_amount": pool.compute_amount
            })
        
        # 全连接层（2层）
        linear_idx = 0
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                self.optical_layers.append({
                    "name": f"FC_{linear_idx}",
                    "layer": layer,
                    "shape": f"({layer.in_features}, {layer.out_features})",
                    "compute_amount": layer.in_features * layer.out_features
                })
                linear_idx += 1
    
    def forward(self, x):
        # 特征提取阶段：光卷积 + 光池化（全部在光域完成）
        x = self.conv1(x)       # [B, 16, 128, 128]
        x = self.pool1(x)       # [B, 16, 64, 64]
        x = F.relu(x)           # [B, 16, 64, 64] — 电计算辅助

        x = self.conv2(x)       # [B, 32, 64, 64]
        x = self.pool2(x)       # [B, 32, 32, 32]
        x = F.relu(x)           # [B, 32, 32, 32]

        x = self.conv3(x)       # [B, 64, 32, 32]
        x = self.pool3(x)       # [B, 64, 16, 16]
        x = F.relu(x)           # [B, 64, 16, 16]

        x = self.conv4(x)       # [B, 128, 16, 16]
        x = self.pool4(x)       # [B, 128, 8, 8]
        x = F.relu(x)           # [B, 128, 8, 8]

        # 展平
        x = x.reshape(x.size(0), -1)  # [B, 8192]

        # 光全连接分类头
        x = self.classifier(x)        # [B, 2]
        return x
