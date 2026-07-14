"""
光计算加速医学影像诊断 — V6 优化版训练脚本
==========================================
改进点：
1. 架构重构：FC0(8192->512) + FC1(512->128) + FC2(128->2) 两级全连接防过拟合
2. BatchNorm + Dropout=0.5 强正则化
3. ReduceLROnPlateau 动态学习率调度
4. 标签平滑 Label Smoothing
5. 数据增强增强：旋转、颜色抖动
6. 测试时增强 TTA
7. 验证集划分 15% 更贴近真实泛化
"""
import time, torch, torch.nn as nn, torch.optim as optim, os, sys, numpy as np
import random
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms

# ========== 超参数 ==========
IMG_SIZE = 128
IN_CHANNELS = 1
NUM_CLASSES = 2
batch_size = 64
epochs = 40
lr = 1e-3
weight_decay = 5e-4          # 增大正则化
dropout_rate = 0.5            # 增大 Dropout
label_smoothing = 0.1         # 标签平滑
TTA_NUM = 5                   # TTA 增强次数

BASE_DIR = '/workspace/Optical_ChestXRay'
DATA_PATH = os.path.join(BASE_DIR, 'data')
LOG_FILE = os.path.join(BASE_DIR, 'output', 'optical_v6_report.txt')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v6.pth')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def init_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== 光计算加速医学影像诊断 V6（优化版）=====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("改进：架构重构+强正则化+LR调度+标签平滑+增强增强+TTA\n\n")


# ========== 增强的数据增强（V6加强）==========
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),               # 新增旋转
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # 新增颜色抖动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ========== V6 优化版模型 ==========
class OpticalConv2d(nn.Module):
    """光模拟卷积模块（与V5保持一致，保证Ro不变）"""
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
    """光计算池化模块（与V5保持一致）"""
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


class OpticalChestXRayV6(nn.Module):
    """
    V6 优化版 — 光计算胸部 X 光肺炎诊断模型

    改进：
    1. FC 层拆分为两级：(8192->512) + (512->128) + (128->2)
    2. 全连接间加 BatchNorm1d + Dropout(0.5) 强正则化
    3. 光卷积/池化层结构与 V5 相同，Ro 保持不变

    维度变化：
    Input:  [B, 1, 128, 128]
    Conv1-4 + Pool1-4: [B, 128, 8, 8]  (与V5完全一致)
    Flatten: [B, 8192]
    FC_0:    [B, 512]  ← 新增中间层
    BN + Dropout + ReLU
    FC_1:    [B, 128]  ← 新增中间层
    BN + Dropout + ReLU
    FC_2:    [B, 2]
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.5):
        super().__init__()

        # ====== 光卷积-池化层（与V5完全一致）=======
        self.conv1 = OpticalConv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = OpticalPool2d(16, kernel_size=2)
        self.conv2 = OpticalConv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = OpticalPool2d(32, kernel_size=2)
        self.conv3 = OpticalConv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = OpticalPool2d(64, kernel_size=2)
        self.conv4 = OpticalConv2d(64, 128, kernel_size=3, padding=1)
        self.pool4 = OpticalPool2d(128, kernel_size=2)

        # ====== V6 优化：两级全连接分类头 ======
        self.fc_input_dim = 128 * 8 * 8  # 8192

        # FC_0: 8192 -> 512（光计算）
        self.fc0 = nn.Linear(self.fc_input_dim, 512)
        self.bn0 = nn.BatchNorm1d(512)    # 电计算：BN辅助稳定训练（不计入Ro）
        self.drop0 = nn.Dropout(dropout_rate)

        # FC_1: 512 -> 128（光计算）
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout_rate)

        # FC_2: 128 -> num_classes（光计算）
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
        # V6 三级全连接
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
        # 光卷积-池化特征提取（全部光计算）
        x = self.conv1(x)
        x = self.pool1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = torch.relu(x)

        x = self.conv4(x)
        x = self.pool4(x)
        x = torch.relu(x)

        # 展平
        x = x.reshape(x.size(0), -1)  # [B, 8192]

        # V6 三级全连接分类头（光计算+辅助电正则化）
        x = self.fc0(x)         # 光计算
        x = self.bn0(x)         # 电辅助
        x = self.drop0(x)
        x = torch.relu(x)

        x = self.fc1(x)         # 光计算
        x = self.bn1(x)         # 电辅助
        x = self.drop1(x)
        x = torch.relu(x)

        x = self.fc2(x)         # 光计算
        return x


# ========== 标签平滑损失 ==========
class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失，防止过拟合"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ========== 训练主函数 ==========
def main():
    init_log()
    log_print(f"设备: {device}")
    log_print(f"V6优化: FC架构重构(8192->512->128->2) | Dropout={dropout_rate} | WD={weight_decay}")
    log_print(f"标签平滑={label_smoothing} | 数据增强+旋转+颜色抖动 | TTA(TTA_NUM) | LR调度ReduceLROnPlateau")
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

    # 验证集划分 15%（更真实反映泛化）
    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 加权采样处理类别不平衡
    train_labels = [train_full.targets[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    log_print(f"训练集分布: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}")
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    log_print(f"训练: {train_size}(加权采样) | 验证: {val_size}(15%) | 测试: {len(test_dataset)}")

    # ====== 初始化模型 ======
    model = OpticalChestXRayV6(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
        dropout_rate=dropout_rate
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"\n参数量: {total_params:,}")

    # 打印光计算层
    log_print("光计算层清单:")
    for info in model.optical_layers:
        log_print(f"  {info['name']:<25} Shape: {info['shape']:<25} 计算量: {info['compute_amount']:>8,}")

    # 光占比计算
    optical_ops = sum(info['compute_amount'] for info in model.optical_layers)
    # 电计算：ReLU x4(特征层) + BN x2 + ReLU x2(FC层) + 5个ReLU+2个BN
    electrical_ops = 16*64*64 + 32*32*32 + 64*16*16 + 128*8*8 + \
                     512 + 128 + 512 + 128  # BN2层 + ReLU2层 + 额外计算
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops if total_ops > 0 else 0
    log_print(f"光计算量: {optical_ops:,} | Ro = {Ro*100:.2f}%\n")

    # ====== 损失函数、优化器、调度器 ======
    # 类别加权
    class_weight_tensor = torch.tensor(
        [1.0/class_counts[0], 1.0/class_counts[1]], dtype=torch.float32
    ).to(device)
    class_weight_tensor = class_weight_tensor / class_weight_tensor.sum() * NUM_CLASSES

    # 标签平滑 + 类别加权
    criterion_base = nn.CrossEntropyLoss(weight=class_weight_tensor)
    criterion_smooth = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # ====== 训练循环 ======
    best_acc = 0.0
    best_val_loss = float('inf')
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

            # 前20个epoch使用标签平滑，后20个使用标准损失
            if epoch < 20:
                loss = criterion_smooth(outputs, lbl)
            else:
                loss = criterion_base(outputs, lbl)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()

            if bidx % 20 == 0:
                print(f"  E{epoch+1:2d}/{epochs} B{bidx:3d} Loss:{loss.item():.4f}")

        # ---- 验证 ----
        model.eval()
        torch.set_grad_enabled(False)
        correct = total = 0
        vloss = 0.0
        class_correct = [0, 0]
        class_total = [0, 0]

        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                outputs = model(img)
                vloss += criterion_base(outputs, lbl).item()
                _, pred = torch.max(outputs, 1)
                total += lbl.size(0)
                correct += (pred == lbl).sum().item()
                for i in range(lbl.size(0)):
                    lb = lbl[i].item()
                    class_total[lb] += 1
                    if pred[i].item() == lb:
                        class_correct[lb] += 1

        acc = correct / total
        avg_vloss = vloss / len(val_loader)

        # 调度器
        scheduler.step(avg_vloss)
        current_lr = optimizer.param_groups[0]['lr']

        # 类别准确率
        cls_str = ' | '.join([
            f"{train_full.classes[i]}: {class_correct[i]}/{class_total[i]}({class_correct[i]/max(class_total[i],1)*100:.1f}%)"
            for i in range(2)
        ])

        log_print(
            f"E[{epoch+1:2d}/{epochs}] "
            f"L:{total_loss/len(train_loader):.4f} "
            f"VL:{avg_vloss:.4f} "
            f"Acc:{acc:.4f}({acc*100:.2f}%) "
            f"[{cls_str}] "
            f"lr:{current_lr:.2e} "
            f"T:{time.time()-start:.0f}s"
        )

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            no_improve_epochs = 0
            log_print(f"  -> 新最佳验证 {best_acc*100:.2f}% (已保存)")
        else:
            no_improve_epochs += 1

        # 早停（如果 15 个 epoch 无提升则停止）
        if no_improve_epochs >= 15:
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
        log_print(f"  {train_full.classes[i]}: {tcc[i]}/{tct[i]} ({tcc[i]/max(tct[i],1)*100:.1f}%)")

    # ====== TTA 测试时增强 ======
    log_print(f"\n===== TTA测试时增强 (TTA_NUM={TTA_NUM}) =====")
    tta_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 用 TTA 重新评估测试集
    tta_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'test'),
        transform=test_transform  # 默认变换，TTA在推理时额外应用
    )

    tc_tta = tt_tta = 0
    with torch.no_grad():
        for idx in range(len(tta_dataset)):
            img_path, lbl = tta_dataset.samples[idx]
            from PIL import Image
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
    log_print(f"\n===== V6 训练完成 =====")


if __name__ == "__main__":
    main()
