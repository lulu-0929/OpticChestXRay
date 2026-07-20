"""
光计算加速的医学影像辅助诊断系统 — 配置文件
==========================================
基于 LTSimulator 光子计算模拟器平台
"""

import time
import torch
from torchvision import transforms
import os

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # Optical_ChestXRay/
LOG_FILE = os.path.join(PROJECT_DIR, "output", "optical_chestxray_report.txt")
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "output", "best_optical_chestxray_v4.pth")
DATA_PATH = os.path.join(PROJECT_DIR, "data")

# 确保输出目录存在
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# 超参数
batch_size = 64          # CPU 训练，减小 batch 避免 OOM
epochs = 25              # 更多epoch
lr = 5e-4                # 更小的学习率
weight_decay = 1e-4      # L2正则化
dropout_rate = 0.2       # Dropout比率

# 图像参数
IMG_SIZE = 128           # 输入图像尺寸
IN_CHANNELS = 1          # 灰度输入（X光片转灰度）
NUM_CLASSES = 2          # 二分类：正常 / 肺炎

# 数据增强（适度增强，避免过度扭曲医学图像）
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=IN_CHANNELS),
    transforms.RandomHorizontalFlip(p=0.5),           # X光片左右对称
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])        # 灰度图归一化
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=IN_CHANNELS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ====== Demo 常量（供 app_demo_v2.py 使用）======
TEMPERATURE = 1.5           # 温度校准参数
NORMAL_THRESHOLD = 0.50     # NORMAL 判决阈值（V18 蒸馏最优）
