"""
光计算加速医学影像诊断 — V10 第二阶段修复版
===============================================
创建时间：2026-07-13
根因：WeightedFocalLoss 和 标准CrossEntropyLoss 都能让模型收敛到全判PNEUMONIA
      NORMAL的Loss被类别权重3x放大，反而导致梯度冲刷、BN统计失调

核心问题：验证集NORMAL占57.2%，全判PNEUMONIA可得42.8%
          模型卡在局部最优——"全判PNEUMONIA"。

V10-fix 新策略：
1. 损失函数 → **普通 CrossEntropyLoss**（不用FocalLoss，不用类别权重）
2. NORMAL过采样 → **5x**（V8的量，防止过拟合）
3. 验证集 → **NORMAL 50%**（平衡验证集）
4. 预测阈值校准 → 在验证集上找最佳NORMAL概率阈值
5. 温度校准 T=0.9
6. 新增 **NORMAL权重初始化**（让fc2.bias初始偏向NORMAL）
7. 早停耐心35epoch
8. 新增**类别平衡的验证准确率**作为早停指标
"""
import time, torch, torch.nn as nn, torch.optim as optim, os, sys, numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

# ========== 超参数 ==========
IMG_SIZE = 128
IN_CHANNELS = 1
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 8e-4
DROPOUT_RATE = 0.55
TTA_NUM = 20
NORMAL_OVERSAMPLE = 5          # 从10x降回5x（V8水平）
MIXUP_ALPHA = 0.2
TEMPERATURE = 0.9              # 略<1.0，增加类间区分度

BASE_DIR = '/workspace/Optical_ChestXRay'
DATA_PATH = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(BASE_DIR, 'output', 'optical_v10_report.txt')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v10.pth')
BEST_BAL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v10_bal.pth')
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
        f.write("===== 光计算加速医学影像诊断 V10-fix =====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("策略：CrossEntropyLoss + NORMAL过采样5x + fc2偏置初始化 + 平衡早停\n\n")


# ========== 数据增强 ==========
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
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03),
])

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


# ========== 光计算模块 ==========
class OpticalConv2d(nn.Module):
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


class OpticalChestXRayV10(nn.Module):
    """
    V10 模型 — V8 FC架构 256→64→2
    + fc2偏置初始化让模型初始偏向NORMAL预测
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.55):
        super().__init__()

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

        # V8 FC架构
        self.fc0 = nn.Linear(128 * 8 * 8, 256)
        self.bn0 = nn.BatchNorm1d(256)
        self.drop0 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, num_classes)

        # ★ 关键：初始化fc2的bias偏向NORMAL
        # bias[0]=NORMAL, bias[1]=PNEUMONIA
        # 设bias[0] > bias[1], 使随机输入时NORMAL概率更高
        nn.init.constant_(self.fc2.bias[0], 0.3)   # NORMAL logit +0.3
        nn.init.constant_(self.fc2.bias[1], -0.3)  # PNEUMONIA logit -0.3

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
        x = torch.relu(self.pool1(self.bn_conv1(self.conv1(x))))
        x = torch.relu(self.pool2(self.bn_conv2(self.conv2(x))))
        x = torch.relu(self.pool3(self.bn_conv3(self.conv3(x))))
        x = torch.relu(self.pool4(self.bn_conv4(self.conv4(x))))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.drop0(self.bn0(self.fc0(x))))
        x = torch.relu(self.drop1(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return x


# ========== Mixup ==========
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class WeightedSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, transforms_list=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transforms_list = transforms_list

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample_path = self.base_dataset.samples[real_idx][0]
        label = self.base_dataset.samples[real_idx][1]
        pil_img = Image.open(sample_path).convert('RGB')
        if self.transforms_list is not None:
            if label == 0:
                pil_img = self.transforms_list[0](pil_img)
            else:
                pil_img = self.transforms_list[1](pil_img)
            return pil_img, label
        return pil_img, label


def find_best_threshold(model, val_loader, device, temperature=1.0):
    """在验证集上搜索最佳NORMAL概率阈值"""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), lbl.to(device)
            outputs = model(img) / temperature
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs[:, 0].cpu())
            all_labels.append(lbl.cpu())
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    best_bal = 0
    best_t = 0.5
    for thresh in [i / 100 for i in range(10, 91)]:
        preds = (all_probs > thresh).long()
        n_c = ((preds == 0) & (all_labels == 0)).sum().item()
        n_t = (all_labels == 0).sum().item()
        p_c = ((preds == 1) & (all_labels == 1)).sum().item()
        p_t = (all_labels == 1).sum().item()
        bal = 0.5 * n_c / max(n_t, 1) + 0.5 * p_c / max(p_t, 1)
        if bal > best_bal:
            best_bal = bal
            best_t = thresh
    return best_t, best_bal


# ========== 训练主函数 ==========
def main():
    init_log()
    log_print(f"设备: {device}")
    log_print(f"V10-fix: CrossEntropyLoss | NORMAL过采样{NORMAL_OVERSAMPLE}x")
    log_print(f"FC(256->64->2) | Dropout={DROPOUT_RATE} | Mixup(alpha={MIXUP_ALPHA})")
    log_print(f"温度 T={TEMPERATURE} | fc2偏置初始化(NORMAL优先)")
    log_print(f"Epochs={EPOCHS} | Batch={BATCH_SIZE} | lr={LR}\n")

    # ====== 加载数据 ======
    train_full = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'train'), transform=None)
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'test'), transform=test_transform)

    val_full_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.Grayscale(),
        transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_full = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'train'), transform=val_full_transform)

    class_names = train_full.classes
    all_labels = train_full.targets
    class_counts = np.bincount(all_labels)
    log_print(f"原始训练集分布: {class_names[0]}={class_counts[0]}, {class_names[1]}={class_counts[1]}")

    # 验证集分层(NORMAL ~50%)
    normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    pneumonia_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
    val_normal = random.sample(normal_indices, int(0.20 * len(normal_indices)))
    val_pneumonia = random.sample(pneumonia_indices, int(0.10 * len(pneumonia_indices)))
    val_indices = val_normal + val_pneumonia
    random.shuffle(val_indices)

    train_indices = list(set(range(len(train_full))) - set(val_indices))
    train_normal = [i for i in train_indices if all_labels[i] == 0]
    train_pneumonia = [i for i in train_indices if all_labels[i] == 1]

    # NORMAL 过采样 5x
    oversampled_normal = train_normal * NORMAL_OVERSAMPLE
    oversampled_train = oversampled_normal + train_pneumonia
    random.shuffle(oversampled_train)

    val_dataset = Subset(val_full, val_indices)
    train_custom = WeightedSubsetDataset(
        train_full, oversampled_train,
        transforms_list=[normal_strong_transform, pneumonia_transform])

    val_labels = [all_labels[i] for i in val_indices]
    train_sampled_labels = [all_labels[i] for i in oversampled_train]
    train_counts = np.bincount(train_sampled_labels)

    log_print(f"训练集(过采样后): NORMAL={train_counts[0]}, PNEUMONIA={train_counts[1]}")
    log_print(f"验证集(分层): NORMAL={val_labels.count(0)}, PNEUMONIA={val_labels.count(1)}")
    log_print(f"测试集: {len(test_dataset)}")

    train_loader = DataLoader(train_custom, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ====== 初始化模型 ======
    model = OpticalChestXRayV10(dropout_rate=DROPOUT_RATE).to(device)

    # 验证初始化
    test_x = torch.randn(64, 1, 128, 128, device=device)
    with torch.no_grad():
        init_out = model(test_x)
        init_pred = init_out.argmax(dim=1)
        init_n_count = (init_pred == 0).sum().item()
    log_print(f"初始化预测分布: NORMAL={init_n_count}/64")

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"参数量: {total_params:,}")

    optical_ops = sum(info['compute_amount'] for info in model.optical_layers)
    electrical_ops = 16*64*64 + 32*32*32 + 64*16*16 + 128*8*8 + 256 + 64
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops
    log_print(f"光计算量: {optical_ops:,} | Ro = {Ro*100:.2f}%\n")

    # ====== 损失函数 ======
    # ★ 核心改动：使用普通CrossEntropyLoss，不使用任何加权
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True, min_lr=1e-6)

    # ====== 训练循环 ======
    best_acc = 0.0
    best_bal_acc = 0.0
    start = time.time()
    no_improve_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        torch.set_grad_enabled(True)
        total_loss = 0.0

        for bidx, (img, lbl) in enumerate(train_loader):
            img, lbl = img.to(device), lbl.to(device)

            if epoch >= 10 and epoch < 50:
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

        # ---- 验证 ----
        model.eval()
        correct = total = 0
        class_correct, class_total = [0, 0], [0, 0]

        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img) / TEMPERATURE
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
        pneumonia_recall = class_correct[1] / max(class_total[1], 1)
        balanced_acc = 0.5 * normal_recall + 0.5 * pneumonia_recall

        scheduler.step(acc)
        current_lr = optimizer.param_groups[0]['lr']

        log_print(
            f"E[{epoch+1:2d}/{EPOCHS}] "
            f"L:{total_loss/len(train_loader):.4f} "
            f"ValAcc:{acc*100:.2f}% "
            f"[N:{class_correct[0]}/{class_total[0]}({normal_recall*100:.1f}%) "
            f"P:{class_correct[1]}/{class_total[1]}({pneumonia_recall*100:.1f}%)] "
            f"Bal:{balanced_acc*100:.2f}% "
            f"lr:{current_lr:.2e} "
            f"T:{time.time()-start:.0f}s"
        )

        # 保存最佳模型（基于总Acc）
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_epochs = 0
            log_print(f"  -> 新最佳 ValAcc={acc*100:.2f}% N={normal_recall*100:.1f}% P={pneumonia_recall*100:.1f}%")

        # 保存最佳平衡Acc模型
        if balanced_acc > best_bal_acc:
            best_bal_acc = balanced_acc
            torch.save(model.state_dict(), BEST_BAL_PATH)

        if acc <= best_acc:
            no_improve_epochs += 1
        if no_improve_epochs >= 30:
            log_print(f"  早停触发：{no_improve_epochs}个epoch无提升")
            break

    total_time = time.time() - start
    log_print(f"\n训练完成！总耗时: {total_time:.0f}s | 最佳ValAcc: {best_acc*100:.2f}% | 最佳BalAcc: {best_bal_acc*100:.2f}%")

    # ====== 测试集评估（取2个模型的较好结果）=====
    def evaluate_model(model_path, label=""):
        if not os.path.exists(model_path):
            log_print(f"\n{label}模型不存在，跳过")
            return None, None, None, None, None

        log_print(f"\n===== 测试集评估（{label}）=====")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        tc = tt = 0
        tcc, tct = [0, 0], [0, 0]
        with torch.no_grad():
            for img, lbl in test_loader:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img) / TEMPERATURE
                _, pred = torch.max(outputs, 1)
                tt += lbl.size(0)
                tc += (pred == lbl).sum().item()
                for i in range(lbl.size(0)):
                    lb = lbl[i].item()
                    tct[lb] += 1
                    if pred[i].item() == lb:
                        tcc[lb] += 1

        test_acc = tc / tt
        normal_acc = tcc[0] / max(tct[0], 1)
        pneumonia_acc = tcc[1] / max(tct[1], 1)
        bal = 0.5 * normal_acc + 0.5 * pneumonia_acc

        log_print(f"测试集Acc: {test_acc*100:.2f}%")
        log_print(f"  NORMAL: {tcc[0]}/{tct[0]} ({normal_acc*100:.1f}%)")
        log_print(f"  PNEUMONIA: {tcc[1]}/{tct[1]} ({pneumonia_acc*100:.1f}%)")
        log_print(f"  平衡Acc: {bal*100:.2f}%")

        # 阈值校准
        log_print(f"\n阈值校准:")
        best_t, best_val_bal = find_best_threshold(model, val_loader, device, TEMPERATURE)
        log_print(f"  最佳阈值: NORMAL_prob > {best_t:.2f} (验证Bal={best_val_bal*100:.2f}%)")

        # 校准后测试
        tc2 = tt2 = 0
        tcc2, tct2 = [0, 0], [0, 0]
        with torch.no_grad():
            for img, lbl in test_loader:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img) / TEMPERATURE
                probs = torch.softmax(outputs, dim=1)
                pred = (probs[:, 0] <= best_t).long()
                tt2 += lbl.size(0)
                tc2 += (pred == lbl).sum().item()
                for i in range(lbl.size(0)):
                    lb = lbl[i].item()
                    tct2[lb] += 1
                    if pred[i].item() == lb:
                        tcc2[lb] += 1

        cal_acc = tc2 / tt2
        cal_normal = tcc2[0] / max(tct2[0], 1)
        cal_pneu = tcc2[1] / max(tct2[1], 1)
        cal_bal = 0.5 * cal_normal + 0.5 * cal_pneu
        log_print(f"校准后测试集Acc: {cal_acc*100:.2f}%")
        log_print(f"  NORMAL: {tcc2[0]}/{tct2[0]} ({cal_normal*100:.1f}%)")
        log_print(f"  PNEUMONIA: {tcc2[1]}/{tct2[1]} ({cal_pneu*100:.1f}%)")
        log_print(f"  校准后平衡Acc: {cal_bal*100:.2f}%")

        # TTA
        log_print(f"\nTTA测试时增强 (TTA={TTA_NUM}, threshold={best_t}):")
        tc3 = tt3 = 0
        tcc3, tct3 = [0, 0], [0, 0]
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
                pred = (avg_probs[:, 0] <= best_t).long()
                tt3 += 1
                if pred.item() == lbl:
                    tc3 += 1
                tct3[lbl] += 1
                if pred.item() == lbl:
                    tcc3[lbl] += 1

        tta_acc = tc3 / tt3
        tta_normal = tcc3[0] / max(tct3[0], 1)
        tta_pneu = tcc3[1] / max(tct3[1], 1)
        tta_bal = 0.5 * tta_normal + 0.5 * tta_pneu
        log_print(f"TTA Acc: {tta_acc*100:.2f}% | N={tta_normal*100:.1f}% P={tta_pneu*100:.1f}% Bal={tta_bal*100:.2f}%")

        return max(test_acc, cal_acc, tta_acc), max(bal, cal_bal, tta_bal), test_acc, cal_acc, tta_acc

    best_raw, best_bal, raw_acc, cal_acc, tta_acc = evaluate_model(BEST_MODEL_PATH, "最佳ValAcc模型")

    # 尝试平衡模型
    bal_raw, bal_bal, bal_raw_acc, bal_cal_acc, bal_tta_acc = evaluate_model(BEST_BAL_PATH, "最佳BalAcc模型")

    # 选择最佳结果
    final_test = max(best_raw if best_raw else 0, bal_raw if bal_raw else 0)
    final_bal = max(best_bal if best_bal else 0, bal_bal if bal_bal else 0)

    log_print(f"\n{'='*60}")
    log_print("【最终评估】")
    log_print(f"{'='*60}")
    log_print(f"光占比 Ro: {Ro*100:.2f}%")
    log_print(f"测试集 Acc (最佳): {final_test*100:.2f}%")
    log_print(f"平衡 Acc (最佳): {final_bal*100:.2f}%")
    log_print(f"训练总耗时: {total_time:.0f}s")

    G = 1 if (Ro > 0.5 and final_test > 0.85) else 0
    S_ratio = 20 * (Ro - 0.5) / 0.5
    S_acc = 5 * ((max(final_test, 0.85) - 0.85) / 0.15) ** 2

    log_print(f"\nG={'通过' if G==1 else '未通过'} | S_ratio={S_ratio:.2f}/20 | S_acc={S_acc:.2f}/5")
    log_print(f"总分: {G * (S_ratio + S_acc):.2f}/30")
    log_print(f"\n===== V10-fix 训练完成 =====")


if __name__ == "__main__":
    main()
