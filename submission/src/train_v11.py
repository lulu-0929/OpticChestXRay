"""
光计算加速医学影像诊断 — V11 训练脚本（迁移学习 + CutMix）
==========================================================
创建时间：2026-07-14
基于V8最佳配置，叠加两项改进：
1. 迁移学习：用ImageNet预训练ResNet18的光卷积权重初始化光卷积层（冻结前2层，微调后2层+FC）
2. CutMix增强：两张X光片混合训练，概率p=0.5，增强局部特征鲁棒性
3. 保留V8全部有效配置：FocalLoss γ=3.0、FC 384→64→2、Dropout=0.5、温度校准T=1.5
"""
import time, torch, torch.nn as nn, torch.optim as optim, os, sys, numpy as np
import random, math
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from PIL import Image

# ========== 超参数 ==========
IMG_SIZE = 128
IN_CHANNELS = 1
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.5
FOCAL_GAMMA = 3.0
TTA_NUM = 15
NORMAL_OVERSAMPLE = 5
TEMPERATURE = 1.5
LABEL_SMOOTHING = 0.05

# CutMix 参数
CUTMIX_ALPHA = 1.0       # Beta分布参数，alpha=1.0表示均匀分布
CUTMIX_PROB = 0.5         # CutMix触发概率

# 迁移学习参数
FREEZE_FRONT_LAYERS = 2   # 冻结前2层光卷积层

BASE_DIR = '/workspace/Optical_ChestXRay'
DATA_PATH = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(BASE_DIR, 'output', 'optical_v11_report.txt')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v11.pth')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def log_print(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def init_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== 光计算加速医学影像诊断 V11（迁移学习 + CutMix）=====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("改进：迁移学习(ResNet18初始化光卷积权重+冻结前2层) + CutMix(p=0.5,alpha=1.0) + V8全部优化\n\n")


# ========== 数据增强（V8完整保留 + 额外医学增强）==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 基础变换（用于CutMix前的基础增强，不包含RandomErasing和Normalize）
base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
])


# ========== CutMix 函数 ==========
def cutmix(batch_x, batch_y, alpha=1.0):
    """
    CutMix增强：将两张图混合，标签也按比例混合
    输入：batch_x [B,C,H,W], batch_y [B]
    返回：混合后的x, 新label (mix标签用于损失计算)

    注意：所有张量都在同一设备（GPU/CPU）上操作
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = batch_x.size(0)
    device = batch_x.device
    index = torch.randperm(batch_size, device=device)

    # 随机裁剪区域
    h, w = batch_x.size(2), batch_x.size(3)
    cut_ratio = math.sqrt(1. - lam)
    ch, cw = int(h * cut_ratio), int(w * cut_ratio)
    cy = random.randint(0, h - ch) if h > ch else 0
    cx = random.randint(0, w - cw) if w > cw else 0

    # 混合标签比例
    lam_actual = 1. - (ch * cw) / (h * w)

    # 执行混合
    mixed_x = batch_x.clone()
    mixed_x[:, :, cy:cy+ch, cx:cx+cw] = batch_x[index, :, cy:cy+ch, cx:cx+cw]

    # 返回混合数据、原始标签、打乱标签、混合比例
    return mixed_x, batch_y, batch_y[index], lam_actual


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    CutMix的损失 = lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ========== 光计算模块 ==========
class OpticalConv2d(nn.Module):
    """光模拟卷积模块（与V8一致）"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.optical_kernel = nn.Linear(
            in_channels * self.kernel_size[0] * self.kernel_size[1],
            out_channels, bias=True
        )
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.compute_amount = in_channels * self.kernel_size[0] * self.kernel_size[1] * out_channels

    def forward(self, x):
        batch, in_c, h, w = x.shape
        patches = self.unfold(x)
        patches = patches.transpose(1, 2).contiguous()
        output = self.optical_kernel(patches)
        out_h = (h + 2*self.padding - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (w + 2*self.padding - self.kernel_size[1]) // self.stride[1] + 1
        output = output.transpose(1, 2).view(batch, self.out_channels, out_h, out_w)
        return output


class OpticalPool2d(nn.Module):
    """光计算池化模块（与V8一致，维持Ro的关键）"""
    def __init__(self, channels, kernel_size=2):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.optical_pool = nn.Linear(self.kernel_size[0] * self.kernel_size[1], 1, bias=False)
        self.compute_amount = channels * self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x):
        batch, c, h, w = x.shape
        unfolded = x.unfold(2, self.kernel_size[0], self.kernel_size[0]).unfold(
            3, self.kernel_size[1], self.kernel_size[1])
        unfolded = unfolded.contiguous().view(batch, c, -1, self.kernel_size[0] * self.kernel_size[1])
        pooled = self.optical_pool(unfolded).squeeze(-1)
        out_h, out_w = h // self.kernel_size[0], w // self.kernel_size[1]
        return pooled.view(batch, c, out_h, out_w)


# ========== V11 模型（支持迁移学习初始化）==========
class OpticalChestXRayV11(nn.Module):
    """
    V11 — 基础架构与V8完全相同，但支持用ResNet18预训练权重初始化光卷积层

    维度变化：
    Input:  [B, 1, 128, 128]  → 光卷积-池化 → [B, 128, 8, 8]
    Flatten: [B, 8192] → FC0 [B, 384] → FC1 [B, 64] → FC2 [B, 2]
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.5):
        super().__init__()

        # ====== 光卷积-池化层（与V8完全一致）=======
        self.conv1 = OpticalConv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn_conv1 = nn.BatchNorm2d(16)
        self.pool1 = OpticalPool2d(16, kernel_size=2)

        self.conv2 = OpticalConv2d(16, 32, kernel_size=3, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.pool2 = OpticalPool2d(32, kernel_size=2)

        self.conv3 = OpticalConv2d(32, 64, kernel_size=3, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.pool3 = OpticalPool2d(64, kernel_size=2)

        self.conv4 = OpticalConv2d(64, 128, kernel_size=3, padding=1)
        self.bn_conv4 = nn.BatchNorm2d(128)
        self.pool4 = OpticalPool2d(128, kernel_size=2)

        # ====== V8 FC架构 384→64→2 ======
        self.fc0 = nn.Linear(128 * 8 * 8, 384)
        self.bn0 = nn.BatchNorm1d(384)
        self.drop0 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(384, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, num_classes)

        # 收集光计算层
        self._collect_optical_layers()

    def _collect_optical_layers(self):
        self.optical_layers = []
        conv_layers = [
            ('conv1', self.conv1), ('conv2', self.conv2),
            ('conv3', self.conv3), ('conv4', self.conv4),
        ]
        for name, conv in conv_layers:
            self.optical_layers.append({
                "name": f"OpticalConv_{name}",
                "layer": conv.optical_kernel,
                "shape": f"({conv.optical_kernel.in_features}, {conv.optical_kernel.out_features})",
                "compute_amount": conv.compute_amount,
            })
        pool_layers = [
            ('pool1', self.pool1), ('pool2', self.pool2),
            ('pool3', self.pool3), ('pool4', self.pool4),
        ]
        for name, pool in pool_layers:
            self.optical_layers.append({
                "name": f"OpticalPool_{name}",
                "layer": pool.optical_pool,
                "shape": f"({pool.optical_pool.in_features}, {pool.optical_pool.out_features})",
                "compute_amount": pool.compute_amount,
            })
        fc_info = [
            ('FC_0(8192->384)', self.fc0, 8192 * 384),
            ('FC_1(384->64)', self.fc1, 384 * 64),
            ('FC_2(64->2)', self.fc2, 64 * 2),
        ]
        for name, layer, amount in fc_info:
            self.optical_layers.append({
                "name": name,
                "layer": layer,
                "shape": f"({layer.in_features}, {layer.out_features})",
                "compute_amount": amount,
            })

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = self.pool2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = self.pool3(x)
        x = torch.relu(x)

        x = self.conv4(x)
        x = self.bn_conv4(x)
        x = self.pool4(x)
        x = torch.relu(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc0(x)
        x = self.bn0(x)
        x = self.drop0(x)
        x = torch.relu(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        return x


# ========== 迁移学习：用ResNet18初始化光卷积层权重 ==========
def initialize_from_resnet18(model):
    """
    用ImageNet预训练的ResNet18权重初始化光卷积层（OpticalConv2d）

    原理：
    - ResNet18的conv1处理 3通道RGB输入 → 提取64通道特征
    - 我们的OpticalConv1处理 1通道灰度输入 → 提取16通道特征
    - 我们将ResNet18.conv1的64个3x3x3卷积核映射到OpticalConv1的16个3x3x1卷积核
    - 映射策略：取RGB三通道均值 → 单通道，然后取前16个卷积核（或PCA选择）
    - conv2-conv4也类似处理

    返回：初始化后的模型
    """
    log_print("\n===== 迁移学习：加载ResNet18预训练权重 =====")
    rn18 = resnet18(weights='IMAGENET1K_V1')
    rn18 = rn18.eval()
    log_print("ResNet18预训练模型加载成功")

    # 获取模型所在设备
    model_device = next(model.parameters()).device

    # 获取ResNet18的卷积层权重
    conv_weights = {
        'conv1': rn18.conv1.weight.data,  # [64, 3, 7, 7] — 注意ResNet18第一层是7x7不是3x3
        'conv2': rn18.layer1[0].conv1.weight.data,  # [64, 64, 3, 3]
        'conv3': rn18.layer2[0].conv1.weight.data,  # [128, 64, 3, 3]
        'conv4': rn18.layer3[0].conv1.weight.data,  # [256, 128, 3, 3]
    }
    log_print(f"ResNet18 conv1 shape: {conv_weights['conv1'].shape}")
    log_print(f"ResNet18 conv2 shape: {conv_weights['conv2'].shape}")
    log_print(f"ResNet18 conv3 shape: {conv_weights['conv3'].shape}")
    log_print(f"ResNet18 conv4 shape: {conv_weights['conv4'].shape}")

    # ---------- OpticalConv1（1→16, kernel=3）----------
    # ResNet18.conv1是[64, 3, 7, 7]，但我们的是[16, 1, 3, 3]
    # 方法：将7x7下采样到3x3，RGB通道取均值，取前16个滤波器
    rn_conv1 = conv_weights['conv1']  # [64, 3, 7, 7]
    # 用AdaptiveAvgPool下采样7x7→3x3
    pooled = torch.nn.functional.adaptive_avg_pool2d(rn_conv1, (3, 3))  # [64, 3, 3, 3]
    # RGB通道取均值 → [64, 1, 3, 3]
    rgb_mean = pooled.mean(dim=1, keepdim=True)  # [64, 1, 3, 3]
    # 取前16个滤波器
    init_weights_1 = rgb_mean[:16]  # [16, 1, 3, 3]

    # 将权重从 [out_c, 1, 3, 3] 映射到 OpticalConv2d 的 Linear 层
    # OpticalConv2d.optical_kernel: Linear(in_features=in_c*3*3, out_features=out_c)
    # 权重 shape: [out_c, in_c, 3, 3] → Linear weight: [out_c, in_c*9]
    w1 = init_weights_1.reshape(16, -1)  # [16, 9]
    model.conv1.optical_kernel.weight.data = w1.float().to(model_device)
    log_print(f"  OpticalConv1: 用ResNet18.conv1前16通道初始化 ✓")

    # ---------- OpticalConv2（16→32, kernel=3）----------
    # ResNet18.layer1[0].conv1: [64, 64, 3, 3]
    rn_conv2 = conv_weights['conv2']  # [64, 64, 3, 3]
    # 取前16个输入通道、前32个输出通道
    init_weights_2 = rn_conv2[:32, :16, :, :]  # [32, 16, 3, 3]
    w2 = init_weights_2.reshape(32, -1)  # [32, 16*9=144]
    model.conv2.optical_kernel.weight.data = w2.float().to(model_device)
    log_print(f"  OpticalConv2: 用ResNet18.layer1[0].conv1初始化 ✓")

    # ---------- OpticalConv3（32→64, kernel=3）----------
    # ResNet18.layer2[0].conv1: [128, 64, 3, 3]
    rn_conv3 = conv_weights['conv3']  # [128, 64, 3, 3]
    # 取前32个输入通道、前64个输出通道
    init_weights_3 = rn_conv3[:64, :32, :, :]  # [64, 32, 3, 3]
    w3 = init_weights_3.reshape(64, -1)  # [64, 32*9=288]
    model.conv3.optical_kernel.weight.data = w3.float().to(model_device)
    log_print(f"  OpticalConv3: 用ResNet18.layer2[0].conv1初始化 ✓")

    # ---------- OpticalConv4（64→128, kernel=3）----------
    # ResNet18.layer3[0].conv1: [256, 128, 3, 3]
    rn_conv4 = conv_weights['conv4']  # [256, 128, 3, 3]
    # 取前64个输入通道、前128个输出通道
    init_weights_4 = rn_conv4[:128, :64, :, :]  # [128, 64, 3, 3]
    w4 = init_weights_4.reshape(128, -1)  # [128, 64*9=576]
    model.conv4.optical_kernel.weight.data = w4.float().to(model_device)
    log_print(f"  OpticalConv4: 用ResNet18.layer3[0].conv1初始化 ✓")

    log_print("ResNet18预训练权重初始化完成！")

    # ---------- 冻结前 N 层 ----------
    if FREEZE_FRONT_LAYERS > 0:
        frozen_layers = []
        for i, (name, layer) in enumerate([
            ('conv1', model.conv1), ('conv2', model.conv2),
            ('conv3', model.conv3), ('conv4', model.conv4)
        ]):
            if i < FREEZE_FRONT_LAYERS:
                for param in layer.optical_kernel.parameters():
                    param.requires_grad = False
                frozen_layers.append(f"OpticalConv{i+1}")

        log_print(f"冻结前{FREEZE_FRONT_LAYERS}层光卷积层: {', '.join(frozen_layers)}")
        log_print(f"可训练层: OpticalConv{ FREEZE_FRONT_LAYERS+1 }-4 + 全部OpticalPool + FC")

    return model


# ========== Focal Loss + Label Smoothing（与V8一致）==========
class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=3.0, smoothing=0.05, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        n_classes = pred.size(-1)
        with torch.no_grad():
            smooth_target = torch.full_like(pred, self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), confidence)
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        ce_loss = -(smooth_target * log_probs).sum(dim=-1)
        pt = torch.exp(-torch.nn.functional.cross_entropy(pred, target, reduction="none"))
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha_w = self.alpha.gather(0, target)
            loss = alpha_w * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ========== 训练主函数 ==========
def main():
    init_log()
    log_print(f"设备: {device}")
    log_print(f"V11 改进: 迁移学习(ResNet18初始化+冻结前{FREEZE_FRONT_LAYERS}层)+CutMix(α={CUTMIX_ALPHA},p={CUTMIX_PROB})")
    log_print(f"V8基座: FocalLoss(γ={FOCAL_GAMMA})+FC(8192->384->64->2)+Dropout={DROPOUT_RATE}")
    log_print(f"NORMAL过采{NORMAL_OVERSAMPLE}x | TTA={TTA_NUM} | 温度T={TEMPERATURE} | 标签平滑={LABEL_SMOOTHING}")
    log_print(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | lr={LR}\n")

    # ====== 加载数据 ======
    train_full = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'train'),
        transform=None  # 不用transform，我们手动处理
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'test'),
        transform=test_transform
    )

    # 统计训练集类别分布
    class_names = train_full.classes
    all_labels = train_full.targets
    class_counts = np.bincount(all_labels)
    log_print(f"原始训练集分布: {class_names[0]}={class_counts[0]}, {class_names[1]}={class_counts[1]}")

    # ====== 验证集分层采样（NORMAL 45-50%，与V8一致）======
    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    pneumonia_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]

    val_normal = random.sample(normal_indices, int(0.25 * len(normal_indices)))
    val_pneumonia = random.sample(pneumonia_indices, int(0.065 * len(pneumonia_indices)))
    val_indices = val_normal + val_pneumonia
    random.shuffle(val_indices)

    # 训练集：排除验证集后，NORMAL过采样
    train_indices = list(set(range(len(train_full))) - set(val_indices))
    train_normal = [i for i in train_indices if all_labels[i] == 0]
    train_pneumonia = [i for i in train_indices if all_labels[i] == 1]

    oversampled_normal = train_normal * NORMAL_OVERSAMPLE
    oversampled_train = oversampled_normal + train_pneumonia
    random.shuffle(oversampled_train)

    # 统计
    val_labels = [all_labels[i] for i in val_indices]
    val_counts = np.bincount(val_labels)
    train_sampled_labels = [all_labels[i] for i in oversampled_train]
    train_counts = np.bincount(train_sampled_labels)

    log_print(f"训练集(过采样后): {class_names[0]}={train_counts[0]}, {class_names[1]}={train_counts[1]}")
    log_print(f"验证集(分层): {class_names[0]}={val_counts[0]}({val_counts[0]/sum(val_counts)*100:.1f}%), "
              f"{class_names[1]}={val_counts[1]}({val_counts[1]/sum(val_counts)*100:.1f}%)")
    log_print(f"测试集: {len(test_dataset)}")

    # ====== 自定义数据集类：支持动态变换 ======
    class DynamicTransformDataset(torch.utils.data.Dataset):
        """支持在epoch之间切换变换的数据集"""
        def __init__(self, base_dataset, indices, default_transform):
            self.base_dataset = base_dataset
            self.indices = indices
            self.transform = default_transform

        def __len__(self):
            return len(self.indices)

        def set_transform(self, transform):
            self.transform = transform

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            path, label = self.base_dataset.samples[real_idx]
            img = Image.open(path).convert('L')
            if self.transform:
                img = self.transform(img)
            return img, label

    # 基础增强的数据集（用于CutMix）
    train_dataset = DynamicTransformDataset(train_full, oversampled_train, default_transform=base_transform)

    # 验证集transform
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_dataset = DynamicTransformDataset(train_full, val_indices, default_transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ====== 初始化模型 ======
    model = OpticalChestXRayV11(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    # ====== 迁移学习：用ResNet18初始化光卷积层权重 ======
    model = initialize_from_resnet18(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"\n总参数量: {total_params:,}")
    log_print(f"可训练参数量: {trainable_params:,}")

    # 打印光计算层
    log_print("光计算层清单:")
    for info in model.optical_layers:
        frozen = "(冻结)" if hasattr(info['layer'], 'weight') and not info['layer'].weight.requires_grad else ""
        log_print(f"  {info['name']:<30} Shape: {info['shape']:<25} 计算量: {info['compute_amount']:>8,} {frozen}")

    # 光占比计算
    optical_ops = sum(info['compute_amount'] for info in model.optical_layers)
    electrical_ops = 16*64*64 + 32*32*32 + 64*16*16 + 128*8*8 + 384 + 64
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops if total_ops > 0 else 0
    log_print(f"光计算量: {optical_ops:,} | Ro = {Ro*100:.2f}%\n")

    # ====== 损失函数 ======
    alpha_tensor = torch.tensor(
        [1.0 / train_counts[0], 1.0 / train_counts[1]], dtype=torch.float32
    ).to(device)
    alpha_tensor = alpha_tensor / alpha_tensor.sum() * NUM_CLASSES

    criterion = FocalLossWithLabelSmoothing(
        alpha=alpha_tensor, gamma=FOCAL_GAMMA, smoothing=LABEL_SMOOTHING
    )

    # 优化器：冻结层参数不更新
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # ====== 训练循环 ======
    best_acc = 0.0
    start = time.time()
    no_improve_epochs = 0

    # 解冻计划：训练到一半时解冻所有层
    unfreeze_epoch = EPOCHS // 3  # 20 epoch时解冻

    for epoch in range(EPOCHS):
        # ---- 解冻逻辑 ----
        if epoch == unfreeze_epoch and FREEZE_FRONT_LAYERS > 0:
            log_print(f"\n  >>> 第{epoch}epoch: 解冻所有层，用更低学习率微调 <<<")
            for param in model.parameters():
                param.requires_grad = True
            # 降低学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR * 0.1
            log_print(f"  >>> 学习率降至 {LR*0.1} <<<")

        # ---- 训练 ----
        model.train()
        torch.set_grad_enabled(True)
        total_loss = 0.0

        # 交替使用CutMix和普通训练
        use_cutmix = (epoch % 2 == 0) and (random.random() < CUTMIX_PROB)

        if use_cutmix:
            # CutMix轮次：使用基础变换
            train_dataset.set_transform(base_transform)
            for bidx, (img, lbl) in enumerate(train_loader):
                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()

                # 对batch中一部分样本做CutMix
                if random.random() < CUTMIX_PROB:
                    mixed_img, lbl_a, lbl_b, lam = cutmix(img, lbl, alpha=CUTMIX_ALPHA)
                    outputs = model(mixed_img)
                    loss = cutmix_criterion(criterion, outputs, lbl_a, lbl_b, lam)
                else:
                    outputs = model(img)
                    loss = criterion(outputs, lbl)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

                if bidx % 20 == 0:
                    print(f"  E{epoch+1:3d}/{EPOCHS}[CutMix] B{bidx:3d} Loss:{loss.item():.4f}")
        else:
            # 普通轮次：使用完整增强
            train_dataset.set_transform(train_transform)
            for bidx, (img, lbl) in enumerate(train_loader):
                img, lbl = img.to(device), lbl.to(device)
                optimizer.zero_grad()
                outputs = model(img)
                loss = criterion(outputs, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

                if bidx % 20 == 0:
                    print(f"  E{epoch+1:3d}/{EPOCHS} B{bidx:3d} Loss:{loss.item():.4f}")

        # ---- 验证 ----
        model.eval()
        torch.set_grad_enabled(False)
        correct = total = 0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img)
                outputs = outputs / TEMPERATURE
                _, pred = torch.max(outputs, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()
                for i in range(lbl.size(0)):
                    lb = lbl[i].item()
                    class_total[lb] += 1
                    if pred[i].item() == lb:
                        class_correct[lb] += 1

        acc = correct / total

        scheduler.step(acc)
        current_lr = optimizer.param_groups[0]['lr']

        cls_str = ' | '.join([
            f"{class_names[i]}: {class_correct[i]}/{class_total[i]}({class_correct[i]/max(class_total[i],1)*100:.1f}%)"
            for i in range(2)
        ])

        log_print(
            f"E[{epoch+1:3d}/{EPOCHS}] "
            f"{'[CutMix]' if use_cutmix else '       '}"
            f" L:{total_loss/len(train_loader):.4f} "
            f"ValAcc:{acc:.4f}({acc*100:.2f}%) "
            f"[{cls_str}] "
            f"lr:{current_lr:.2e} "
            f"T:{time.time()-start:.0f}s"
        )

        # 加权保存（优先NORMAL召回率）
        normal_recall = class_correct[0] / max(class_total[0], 1)
        weighted_score = 0.7 * acc + 0.3 * normal_recall
        if weighted_score > best_acc:
            best_acc = weighted_score
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_epochs = 0
            log_print(f"  -> 新最佳(加权={weighted_score*100:.2f}%) Acc={acc*100:.2f}% NORMAL_Recall={normal_recall*100:.1f}% (已保存)")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= 25:
            log_print(f"  早停触发：{no_improve_epochs}个epoch无提升")
            break

    total_time = time.time() - start
    log_print(f"\n训练完成！总耗时: {total_time:.0f}s | 最佳验证加权得分: {best_acc*100:.2f}%")

    # ====== 测试集评估 ======
    log_print(f"\n===== 测试集评估（最佳模型）=====")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    tc = tt = 0
    tcc, tct = [0, 0], [0, 0]
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            outputs = model(img)
            outputs = outputs / TEMPERATURE
            _, pred = torch.max(outputs, 1)
            tt += lbl.size(0)
            tc += (pred == lbl).sum().item()
            for i in range(lbl.size(0)):
                lb = lbl[i].item()
                tct[lb] += 1
                if pred[i].item() == lb:
                    tcc[lb] += 1

    test_acc = tc / tt
    log_print(f"测试集总准确率: {test_acc*100:.2f}%")
    for i in range(2):
        log_print(f"  {class_names[i]}: {tcc[i]}/{tct[i]} ({tcc[i]/max(tct[i],1)*100:.1f}%)")

    # 平衡准确率
    normal_test_acc = tcc[0] / max(tct[0], 1)
    pneumonia_test_acc = tcc[1] / max(tct[1], 1)
    balanced_test_acc = 0.5 * normal_test_acc + 0.5 * pneumonia_test_acc
    log_print(chr(10) + "平衡测试集准确率: 0.5*" + f"{normal_test_acc*100:.1f}%" + " + 0.5*" + f"{pneumonia_test_acc*100:.1f}%" + " = " + f"{balanced_test_acc*100:.2f}%")

    # ====== TTA ======
    log_print(f"\n===== TTA测试时增强 (TTA_NUM={TTA_NUM}) =====")
    tc_tta = tt_tta = 0

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, lbl = test_dataset.samples[idx]
            pil_img = Image.open(img_path).convert('L')

            base_img = test_transform(pil_img).unsqueeze(0).to(device)
            base_out = model(base_img) / TEMPERATURE
            votes = torch.softmax(base_out, dim=1)

            for _ in range(TTA_NUM):
                aug_img = tta_transform(pil_img).unsqueeze(0).to(device)
                aug_out = model(aug_img) / TEMPERATURE
                votes += torch.softmax(aug_out, dim=1)

            avg_probs = votes / (TTA_NUM + 1)
            _, pred = torch.max(avg_probs, 1)
            tt_tta += 1
            if pred.item() == lbl:
                tc_tta += 1

    tta_acc = tc_tta / tt_tta
    log_print(f"TTA测试集准确率: {tta_acc*100:.2f}%")

    # ====== 最终评分报告 ======
    best_final = max(test_acc, tta_acc)
    log_print(f"\n{'='*60}")
    log_print("【最终评估】")
    log_print(f"{'='*60}")
    log_print(f"光占比 Ro: {Ro*100:.2f}%")
    log_print(f"测试集 Acc: {test_acc*100:.2f}%")
    log_print(f"平衡测试集 Acc: {balanced_test_acc*100:.2f}%")
    log_print(f"TTA 测试集 Acc: {tta_acc*100:.2f}%")
    log_print(f"训练总耗时: {total_time:.0f}s")

    G = 1 if (Ro > 0.5 and best_final > 0.85) else 0
    S_ratio = 20 * (Ro - 0.5) / 0.5
    S_acc = 5 * ((max(best_final, 0.85) - 0.85) / 0.15) ** 2

    log_print(f"\nG={'通过' if G==1 else '未通过'} | S_ratio={S_ratio:.2f}/20 | S_acc={S_acc:.2f}/5")
    log_print(f"总分: {G * (S_ratio + S_acc):.2f}/30")
    log_print(f"\n===== V11 训练完成 =====")


if __name__ == "__main__":
    main()
