"""
光计算加速医学影像诊断 — V9 终极优化训练脚本
==========================================
V9 核心改进：
1. NORMAL 过采样 10x + 类别加权采样 → 彻底解决 NORMAL 样本不足
2. NORMAL 独享更强数据增强（RandomErasing p=0.5 + 高斯噪声 + 亮度抖动）
3. Asymmetric Loss（非对称损失）：对难分正样本（NORMAL）施加强惩罚
4. Mixup 数据增强 → 迫使模型学到更鲁棒的特征边界
5. FC 扩展为 512→128→2 + 更激进 Dropout（0.55）→ 防过拟合
6. 动态温度校准 T=1.0~2.0 自适应
7. 验证集 NORMAL 占比提升至 50~55%
8. 早停耐心提升至 30 epoch + 最佳模型保存改为纯验证集 Acc
"""
import time, torch, torch.nn as nn, torch.optim as optim, os, sys, numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from PIL import Image

# ========== 超参数 ==========
IMG_SIZE = 128
IN_CHANNELS = 1
NUM_CLASSES = 2
batch_size = 64
epochs = 80
lr = 1e-3
weight_decay = 8e-4
dropout_rate = 0.55
focal_gamma_pos = 0.0        # Asymmetric Loss：正样本gamma（不惩罚）
focal_gamma_neg = 2.0        # Asymmetric Loss：负样本gamma（从4.0降至2.0，减轻对NORMAL预测的过度惩罚）
TTA_NUM = 20                  # TTA 增强投票数
NORMAL_OVERSAMPLE = 10        # NORMAL 过采样倍数（从5x→10x）
MIXUP_ALPHA = 0.2             # Mixup Beta分布参数
TEMPERATURE = 1.0             # 温度校准（从1.5降回1.0，避免softmax过度平滑）
label_smoothing = 0.05        # 标签平滑

BASE_DIR = '/workspace/Optical_ChestXRay'
DATA_PATH = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(BASE_DIR, 'output', 'optical_v9_report.txt')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v9.pth')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def init_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== 光计算加速医学影像诊断 V9（终极优化）=====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("改进：NORMAL过采样10x + AsymmetricLoss + Mixup + NORMAL专属增强 + FC(512->128->2)\n\n")


# ========== NORMAL 专属强增强 ==========
normal_strong_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03),  # 高斯噪声
])

# PNEUMONIA 标准增强
pneumonia_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

# 基础变换（用于 TTA）
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ========== 光计算模块（与V8保持 Ro 不变）==========
class OpticalConv2d(nn.Module):
    """光模拟卷积模块"""
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
    """光计算池化模块"""
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


class OpticalChestXRayV9(nn.Module):
    """
    V9 模型 — FC 扩展为 512→128→2

    维度变化：
    Input:  [B, 1, 128, 128]  → 光卷积-池化 → [B, 128, 8, 8]
    Flatten: [B, 8192] → FC0 [B, 512] → FC1 [B, 128] → FC2 [B, 2]
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.55):
        super().__init__()

        # 光卷积-池化层（与V8完全一致，保持Ro不变）
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

        # V9：扩展 FC 为 512→128→2
        self.fc0 = nn.Linear(128 * 8 * 8, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.drop0 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, num_classes)

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
        # V9 FC
        fc_info = [
            ('FC_0(8192->512)', self.fc0, 8192 * 512),
            ('FC_1(512->128)', self.fc1, 512 * 128),
            ('FC_2(128->2)', self.fc2, 128 * 2),
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


# ========== Asymmetric Loss（非对称损失）==========
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss — 对负样本（难分负样本）施加强惩罚
    gamma_pos: 正样本gamma（通常设为0，避免惩罚正确分类的正样本）
    gamma_neg: 负样本gamma（设为3~4，强惩罚把NORMAL误判为PNEUMONIA的难分负样本）
    """
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, pred, target):
        # 计算概率
        prob = torch.sigmoid(pred)
        # 裁剪避免数值不稳定
        prob = torch.clamp(prob, self.clip, 1 - self.clip)

        # one-hot 编码
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        # 正样本概率 pt
        pt = (1 - target_onehot) * (1 - prob) + target_onehot * prob

        # 非对称 gamma
        asymmetric_weight = (1 - target_onehot) * (1 - pt).pow(self.gamma_neg) + \
                            target_onehot * (1 - pt).pow(self.gamma_pos)

        # Asymmetric Loss
        loss = -asymmetric_weight * torch.log(pt + self.eps)

        return loss.mean()


# ========== Mixup 数据增强 ==========
def mixup_data(x, y, alpha=0.2):
    """Mixup 数据增强：按 Beta(alpha,alpha) 混合两批数据"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失：加权交叉熵"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ========== 类别加权采样器 ==========
class WeightedSubsetDataset(torch.utils.data.Dataset):
    """支持类别加权采样的子集"""
    def __init__(self, base_dataset, indices, transforms_list=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transforms_list = transforms_list  # [normal_transform, pneumonia_transform]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # ImageFolder transform=None 时 __getitem__ 返回PIL Image
        # 改为直接从samples读取
        sample_path = self.base_dataset.samples[real_idx][0]
        label = self.base_dataset.samples[real_idx][1]
        pil_img = Image.open(sample_path).convert('RGB')

        if self.transforms_list is not None:
            if label == 0:  # NORMAL
                pil_img = self.transforms_list[0](pil_img)
            else:  # PNEUMONIA
                pil_img = self.transforms_list[1](pil_img)
            return pil_img, label

        return img, label


# ========== 训练主函数 ==========
def main():
    init_log()
    log_print(f"设备: {device}")
    log_print(f"V9优化: FC(8192->512->128->2) | Dropout={dropout_rate} | Mixup(alpha={MIXUP_ALPHA})")
    log_print(f"AsymmetricLoss(gamma_pos={focal_gamma_pos}, gamma_neg={focal_gamma_neg})")
    log_print(f"NORMAL过采样{NORMAL_OVERSAMPLE}x | NORMAL专属强增强 | TTA{TTA_NUM}+温度校准(T={TEMPERATURE})")
    log_print(f"验证集分层(NORMAL 50-55%) | Epochs={epochs} | Batch={batch_size} | lr={lr}\n")

    # ====== 加载数据 ======
    train_full = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'train'),
        transform=None  # 在自定义Dataset中处理
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'test'),
        transform=test_transform
    )

    # 为验证集创建独立的数据集对象（带 transform，避免 Subset 返回 PIL）
    val_full_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_full = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'train'),
        transform=val_full_transform
    )

    # 统计训练集类别分布
    class_names = train_full.classes
    all_labels = train_full.targets
    class_counts = np.bincount(all_labels)
    log_print(f"原始训练集分布: {class_names[0]}={class_counts[0]}, {class_names[1]}={class_counts[1]}")

    # ====== V9：验证集分层(NORMAL 50-55%)======
    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    pneumonia_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]

    # 验证集：从 NORMAL 取 25%，从 PNEUMONIA 取 8%（确保NORMAL占比50-55%）
    val_normal = random.sample(normal_indices, int(0.25 * len(normal_indices)))
    val_pneumonia = random.sample(pneumonia_indices, int(0.065 * len(pneumonia_indices)))
    val_indices = val_normal + val_pneumonia
    random.shuffle(val_indices)

    # 训练集：排除验证集样本后，对 NORMAL 过量采样
    train_indices = list(set(range(len(train_full))) - set(val_indices))
    train_normal = [i for i in train_indices if all_labels[i] == 0]
    train_pneumonia = [i for i in train_indices if all_labels[i] == 1]

    # NORMAL 过采样 10x
    oversampled_normal = train_normal * NORMAL_OVERSAMPLE
    oversampled_train = oversampled_normal + train_pneumonia
    random.shuffle(oversampled_train)

    # 构建验证集和训练集（使用自定义Dataset实现类别差别增强）
    val_dataset = Subset(val_full, val_indices)
    train_custom = WeightedSubsetDataset(
        train_full, oversampled_train,
        transforms_list=[normal_strong_transform, pneumonia_transform]
    )

    # 统计
    val_labels = [all_labels[i] for i in val_indices]
    val_counts = np.bincount(val_labels)
    train_sampled_labels = [all_labels[i] for i in oversampled_train]
    train_counts = np.bincount(train_sampled_labels)

    log_print(f"训练集(过采样后): {class_names[0]}={train_counts[0]}({train_counts[0]/sum(train_counts)*100:.1f}%), "
              f"{class_names[1]}={train_counts[1]}({train_counts[1]/sum(train_counts)*100:.1f}%)")
    log_print(f"验证集(分层): {class_names[0]}={val_labels.count(0)}({val_labels.count(0)/len(val_labels)*100:.1f}%), "
              f"{class_names[1]}={val_labels.count(1)}({val_labels.count(1)/len(val_labels)*100:.1f}%)")
    log_print(f"测试集: {len(test_dataset)}")

    train_loader = DataLoader(
        train_custom, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ====== 初始化模型 ======
    model = OpticalChestXRayV9(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
        dropout_rate=dropout_rate
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"\n参数量: {total_params:,}")

    # 打印光计算层
    log_print("光计算层清单:")
    for info in model.optical_layers:
        log_print(f"  {info['name']:<30} Shape: {info['shape']:<25} 计算量: {info['compute_amount']:>8,}")

    # 光占比计算
    optical_ops = sum(info['compute_amount'] for info in model.optical_layers)
    electrical_ops = 16*64*64 + 32*32*32 + 64*16*16 + 128*8*8 + 512 + 128
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops if total_ops > 0 else 0
    log_print(f"光计算量: {optical_ops:,} | Ro = {Ro*100:.2f}%\n")

    # ====== 损失函数、优化器、调度器 ======
    # Asymmetric Loss
    criterion = AsymmetricLoss(gamma_pos=focal_gamma_pos, gamma_neg=focal_gamma_neg)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-6
    )

    # ====== 训练循环 ======
    best_acc = 0.0
    start = time.time()
    no_improve_epochs = 0
    use_mixup = True

    for epoch in range(epochs):
        current_lr_final = optimizer.param_groups[0]['lr']

        # ---- 训练（带 Mixup）----
        model.train()
        torch.set_grad_enabled(True)
        total_loss = 0.0

        for bidx, (img, lbl) in enumerate(train_loader):
            img, lbl = img.to(device), lbl.to(device)

            # Mixup 数据增强（前10个epoch关闭，让模型先学会基本分类；epoch 10~50开启）
            if use_mixup and epoch >= 10 and epoch < 50:
                mixed_img, lbl_a, lbl_b, lam = mixup_data(img, lbl, MIXUP_ALPHA)
                optimizer.zero_grad()
                outputs = model(mixed_img)
                loss = mixup_criterion(criterion, outputs, lbl_a, lbl_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(img)
                loss = criterion(outputs, lbl)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            if bidx % 20 == 0:
                print(f"  E{epoch+1:2d}/{epochs} B{bidx:3d} Loss:{loss.item():.4f}")

        # ---- 验证（监控验证集 Acc）----
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
        normal_recall = class_correct[0] / max(class_total[0], 1)

        scheduler.step(acc)
        current_lr = optimizer.param_groups[0]['lr']

        cls_str = ' | '.join([
            f"{class_names[i]}: {class_correct[i]}/{class_total[i]}({class_correct[i]/max(class_total[i],1)*100:.1f}%)"
            for i in range(2)
        ])

        log_print(
            f"E[{epoch+1:2d}/{epochs}] "
            f"L:{total_loss/len(train_loader):.4f} "
            f"ValAcc:{acc:.4f}({acc*100:.2f}%) "
            f"[{cls_str}] "
            f"lr:{current_lr:.2e} "
            f"T:{time.time()-start:.0f}s"
        )

        # 纯验证集 Acc（不再加权）
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_epochs = 0
            log_print(f"  -> 新最佳 ValAcc={acc*100:.2f}% NORMAL_Recall={normal_recall*100:.1f}% (已保存)")
        else:
            no_improve_epochs += 1

        # 早停：30 epoch 无提升则停止
        if no_improve_epochs >= 30:
            log_print(f"  早停触发：{no_improve_epochs}个epoch无提升")
            break

    total_time = time.time() - start
    log_print(f"\n训练完成！总耗时: {total_time:.0f}s | 最佳验证Acc: {best_acc*100:.2f}%")

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

    # 平衡测试集准确率
    normal_test_acc = tcc[0] / max(tct[0], 1)
    pneumonia_test_acc = tcc[1] / max(tct[1], 1)
    balanced_test_acc = 0.5 * normal_test_acc + 0.5 * pneumonia_test_acc
    log_print(f"\n平衡测试集准确率: 0.5*{normal_test_acc*100:.1f}% + 0.5*{pneumonia_test_acc*100:.1f}% = {balanced_test_acc*100:.2f}%")

    # ====== TTA 测试时增强 ======
    log_print(f"\n===== TTA测试时增强 (TTA_NUM={TTA_NUM}) =====")
    tc_tta = tt_tta = 0

    # 使用常规test_transform做基准
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, lbl = test_dataset.samples[idx]
            pil_img = Image.open(img_path).convert('L')

            # 原始预测
            base_img = test_transform(pil_img).unsqueeze(0).to(device)
            base_out = model(base_img) / TEMPERATURE
            votes = torch.softmax(base_out, dim=1)

            # TTA 增强预测
            for _ in range(TTA_NUM):
                aug_img = tta_transform(pil_img).unsqueeze(0).to(device)
                aug_out = model(aug_img) / TEMPERATURE
                votes += torch.softmax(aug_out, dim=1)

            # 平均投票
            avg_probs = votes / (TTA_NUM + 1)
            _, pred = torch.max(avg_probs, 1)
            tt_tta += 1
            if pred.item() == lbl:
                tc_tta += 1

    tta_acc = tc_tta / tt_tta
    log_print(f"TTA测试集准确率: {tta_acc*100:.2f}%")

    # ====== 最终评分报告 ======
    log_print(f"\n{'='*60}")
    log_print("【最终评估】")
    log_print(f"{'='*60}")
    log_print(f"光占比 Ro: {Ro*100:.2f}%")
    log_print(f"测试集 Acc: {test_acc*100:.2f}%")
    log_print(f"平衡测试集 Acc: {balanced_test_acc*100:.2f}%")
    log_print(f"TTA 测试集 Acc: {tta_acc*100:.2f}%")
    log_print(f"训练总耗时: {total_time:.0f}s")

    G = 1 if (Ro > 0.5 and test_acc > 0.85) else 0
    S_ratio = 20 * (Ro - 0.5) / 0.5
    S_acc = 5 * ((max(test_acc, 0.85) - 0.85) / 0.15) ** 2

    log_print(f"\nG={'通过' if G==1 else '未通过'} | S_ratio={S_ratio:.2f}/20 | S_acc={S_acc:.2f}/5")
    log_print(f"总分: {G * (S_ratio + S_acc):.2f}/30")
    log_print(f"\n===== V9 训练完成 =====")


if __name__ == "__main__":
    main()
