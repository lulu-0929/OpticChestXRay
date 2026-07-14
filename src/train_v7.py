"""
光计算加速医学影像诊断 — V7 针对性优化训练脚本
==========================================
V7 核心改进：
1. Focal Loss + 类别加权 → 强制关注难分的 NORMAL 样本
2. FC 参数精简 (8192->256->64->2) → 防过拟合
3. 验证集分层采样 → NORMAL 占比不低于 35%，消除验证集偏差
4. NORMAL 类别过采样 3x + 数据增强翻倍
5. 早停耐心值 20 epoch + 更激进 LR 衰减
6. TTA 增强投票数提升至 10
"""
import time, torch, torch.nn as nn, torch.optim as optim, os, sys, numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

# ========== 超参数 ==========
IMG_SIZE = 128
IN_CHANNELS = 1
NUM_CLASSES = 2
batch_size = 64
epochs = 50
lr = 1e-3
weight_decay = 1e-3           # 加强正则化
dropout_rate = 0.5            # 保持 Dropout
focal_gamma = 2.0             # Focal Loss gamma 参数
TTA_NUM = 10                  # TTA 增强投票数
NORMAL_OVERSAMPLE = 3         # NORMAL 过采样倍数

BASE_DIR = '/workspace/Optical_ChestXRay'
DATA_PATH = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(BASE_DIR, 'output', 'optical_v7_report.txt')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v7.pth')
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
        f.write("===== 光计算加速医学影像诊断 V7（针对性优化）=====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("改进：FocalLoss+FC精简+验证集分层+NORMAL过采样3x+TTA10\n\n")


# ========== 强数据增强（NORMAL 过采样配合更强增强）==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),               # 更大旋转
    transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 基础变换（用于 TTA）
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ========== 光计算模块 ==========
class OpticalConv2d(nn.Module):
    """光模拟卷积模块（与 V5/V6 保持一致）"""
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


class OpticalChestXRayV7(nn.Module):
    """
    V7 优化版 — 精简 FC + 强防过拟合

    改进：
    1. FC 参数精简：(8192->256) + (256->64) + (64->2)
    2. 参数量仅为 V6 的一半左右
    3. 更强的 Dropout + BatchNorm 防过拟合

    维度变化：
    Input:  [B, 1, 128, 128]  → 光卷积-池化 → [B, 128, 8, 8]
    Flatten: [B, 8192] → FC0 [B, 256] → FC1 [B, 64] → FC2 [B, 2]
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.5):
        super().__init__()

        # ====== 光卷积-池化层（与V5/V6完全一致，保持Ro）=======
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

        # ====== V7：精简三级全连接 ======
        self.fc0 = nn.Linear(128 * 8 * 8, 256)        # 8192 -> 256
        self.bn0 = nn.BatchNorm1d(256)
        self.drop0 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256, 64)                 # 256 -> 64
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, num_classes)         # 64 -> 2

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
        # V7 精简 FC
        fc_info = [
            ('FC_0(8192->256)', self.fc0, 8192 * 256),
            ('FC_1(256->64)', self.fc1, 256 * 64),
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
        # 光卷积-池化（全部光计算，BN 和 ReLU 是电辅助不计入 Ro）
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

        # 展平
        x = x.reshape(x.size(0), -1)  # [B, 8192]

        # V7 精简全连接
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


# ========== Focal Loss ==========
class FocalLoss(nn.Module):
    """
    Focal Loss — 强制关注难分类样本
    适用场景：严重类别不平衡 + NORMAL 难分
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha          # 类别权重 [alpha_0, alpha_1]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        ce_loss = torch.nn.functional.cross_entropy(pred, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)    # 预测正确概率
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ========== 训练主函数 ==========
def main():
    init_log()
    log_print(f"设备: {device}")
    log_print(f"V7优化: FC精简(8192->256->64->2) | Dropout={dropout_rate} | WD={weight_decay}")
    log_print(f"FocalLoss(gamma={focal_gamma}) | NORMAL过采样{NORMAL_OVERSAMPLE}x | TTA(TTA_NUM) | 验证集分层")
    log_print(f"Epochs={epochs} | Batch={batch_size} | lr={lr}\n")

    # ====== 加载数据 ======
    train_full = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'train'),
        transform=train_transform
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

    # ====== V7：验证集分层采样（保证NORMAL占比≥35%）======
    # 获取每个类别的索引
    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    pneumonia_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]

    # 验证集：从 NORMAL 取 25%，从 PNEUMONIA 取 8%
    val_normal = random.sample(normal_indices, int(0.25 * len(normal_indices)))
    val_pneumonia = random.sample(pneumonia_indices, int(0.08 * len(pneumonia_indices)))
    val_indices = val_normal + val_pneumonia
    random.shuffle(val_indices)

    # 训练集：排除验证集样本后，对 NORMAL 进行过采样
    train_indices = list(set(range(len(train_full))) - set(val_indices))
    train_normal = [i for i in train_indices if all_labels[i] == 0]
    train_pneumonia = [i for i in train_indices if all_labels[i] == 1]

    # NORMAL 过采样
    oversampled_normal = train_normal * NORMAL_OVERSAMPLE
    oversampled_train = oversampled_normal + train_pneumonia
    random.shuffle(oversampled_train)

    # 构建验证集和训练集
    val_dataset = Subset(train_full, val_indices)
    train_dataset = Subset(train_full, oversampled_train)

    # 统计
    val_labels = [all_labels[i] for i in val_indices]
    val_counts = np.bincount(val_labels)
    train_sampled_labels = [all_labels[i] for i in oversampled_train]
    train_counts = np.bincount(train_sampled_labels)

    log_print(f"训练集(过采样后): {class_names[0]}={train_counts[0]}, {class_names[1]}={train_counts[1]}")
    log_print(f"验证集(分层): {class_names[0]}={val_counts[0]}({val_counts[0]/sum(val_counts)*100:.1f}%), "
              f"{class_names[1]}={val_counts[1]}({val_counts[1]/sum(val_counts)*100:.1f}%)")
    log_print(f"测试集: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ====== 初始化模型 ======
    model = OpticalChestXRayV7(
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
    electrical_ops = 16*64*64 + 32*32*32 + 64*16*16 + 128*8*8 + 256 + 64
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops if total_ops > 0 else 0
    log_print(f"光计算量: {optical_ops:,} | Ro = {Ro*100:.2f}%\n")

    # ====== 损失函数、优化器、调度器 ======
    # 类别权重（按NORMAL过采样后比例计算）
    alpha_tensor = torch.tensor(
        [1.0 / train_counts[0], 1.0 / train_counts[1]], dtype=torch.float32
    ).to(device)
    alpha_tensor = alpha_tensor / alpha_tensor.sum() * NUM_CLASSES

    # Focal Loss（gamma=2，强制关注难分样本）
    criterion = FocalLoss(alpha=alpha_tensor, gamma=focal_gamma)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )

    # ====== 训练循环 ======
    best_acc = 0.0
    start = time.time()
    no_improve_epochs = 0

    for epoch in range(epochs):
        # ---- 训练 ----
        model.train()
        torch.set_grad_enabled(True)
        total_loss = 0.0

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
                print(f"  E{epoch+1:2d}/{epochs} B{bidx:3d} Loss:{loss.item():.4f}")

        # ---- 验证（监控验证集Acc）----
        model.eval()
        torch.set_grad_enabled(False)
        correct = total = 0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img)
                _, pred = torch.max(outputs, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()
                for i in range(lbl.size(0)):
                    lb = lbl[i].item()
                    class_total[lb] += 1
                    if pred[i].item() == lb:
                        class_correct[lb] += 1

        acc = correct / total

        # 调度器（监控验证集Acc）
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

        # 保存最佳模型（基于验证集Acc）
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_epochs = 0
            log_print(f"  -> 新最佳验证 {best_acc*100:.2f}% (已保存)")
        else:
            no_improve_epochs += 1

        # 早停：20 epoch 无提升则停止
        if no_improve_epochs >= 20:
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

    # ====== TTA 测试时增强 ======
    log_print(f"\n===== TTA测试时增强 (TTA_NUM={TTA_NUM}) =====")
    tc_tta = tt_tta = 0

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, lbl = test_dataset.samples[idx]
            pil_img = Image.open(img_path).convert('L')

            # 原始预测
            base_img = test_transform(pil_img).unsqueeze(0).to(device)
            base_out = model(base_img)
            votes = torch.softmax(base_out, dim=1)

            # TTA 增强预测
            for _ in range(TTA_NUM):
                aug_img = tta_transform(pil_img).unsqueeze(0).to(device)
                aug_out = model(aug_img)
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
    log_print(f"TTA 测试集 Acc: {tta_acc*100:.2f}%")
    log_print(f"训练总耗时: {total_time:.0f}s")

    G = 1 if (Ro > 0.5 and test_acc > 0.85) else 0
    S_ratio = 20 * (Ro - 0.5) / 0.5
    S_acc = 5 * ((max(test_acc, 0.85) - 0.85) / 0.15) ** 2

    log_print(f"\nG={'通过' if G==1 else '未通过'} | S_ratio={S_ratio:.2f}/20 | S_acc={S_acc:.2f}/5")
    log_print(f"总分: {G * (S_ratio + S_acc):.2f}/30")
    log_print(f"\n===== V7 训练完成 =====")


if __name__ == "__main__":
    main()
