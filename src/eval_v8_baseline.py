"""
光计算加速医学影像诊断 — V8 真实基线重新评估
==============================================
目的：修复 test_transform 中 RandomErasing bug，用已有 best_optical_v8.pth 重新评估真实性能
创建时间：2026-07-14
"""
import time, torch, torch.nn as nn, os, sys, numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image

# ========== 超参数 ==========
IMG_SIZE = 128
IN_CHANNELS = 1
NUM_CLASSES = 2
batch_size = 64
dropout_rate = 0.5
TTA_NUM = 15
TEMPERATURE = 1.5

BASE_DIR = '/workspace/Optical_ChestXRay'
DATA_PATH = os.path.join(BASE_DIR, 'data')
REPORT_FILE = os.path.join(BASE_DIR, 'output', 'optical_v8_baseline_fixed.txt')
BEST_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'best_optical_v8.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_print(msg):
    print(msg)
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ========== 修复后的变换（核心：去掉 test_transform 和 tta_transform 中的 RandomErasing）==========
# 修复要点：
# 1. test_transform：只做尺寸调整、灰度、转Tensor、归一化 → 干净的评估
# 2. tta_transform：去掉RandomErasing，只保留几何增强 → 增强后投票更可靠
# 3. train_transform：保持不变（训练时RandomErasing是合理的数据增强）

# 干净的测试变换（修复bug！）
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 干净的 TTA 变换（修复bug！去掉了RandomErasing）
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.03)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ========== 光计算模块（与V8完全一致）==========
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


class OpticalChestXRayV8(nn.Module):
    """与 V8 完全一致的模型结构"""
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.5):
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

        self.fc0 = nn.Linear(128 * 8 * 8, 384)
        self.bn0 = nn.BatchNorm1d(384)
        self.drop0 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(384, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, num_classes)

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


def main():
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("===== V8 真实基线重新评估（修复test_transform bug）=====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("目的：移除test_transform/tta_transform中的RandomErasing，获取真实性能\n")
        f.write("修复：test_transform: Resize→Grayscale→ToTensor→Normalize（干净评估）\n")
        f.write("修复：tta_transform: 去掉RandomErasing，保留几何增强\n\n")

    log_print(f"设备: {device}")
    log_print(f"温度校准 T={TEMPERATURE}, TTA_NUM={TTA_NUM}")
    log_print(f"权重文件: {BEST_MODEL_PATH}")
    log_print(f"报告文件: {REPORT_FILE}\n")

    # ====== 加载测试集 ======
    test_dataset = datasets.ImageFolder(
        root=os.path.join(DATA_PATH, 'chest_xray', 'test'),
        transform=test_transform
    )
    class_names = test_dataset.classes
    log_print(f"测试集: {len(test_dataset)} 张图片")
    log_print(f"类别: {class_names}")

    # 统计测试集分布
    test_labels = test_dataset.targets
    test_counts = np.bincount(test_labels)
    log_print(f"测试集分布: NORMAL={test_counts[0]}, PNEUMONIA={test_counts[1]}\n")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ====== 初始化模型并加载V8权重 ======
    model = OpticalChestXRayV8(
        in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
        dropout_rate=dropout_rate
    ).to(device)

    # 加载已有权重
    if not os.path.exists(BEST_MODEL_PATH):
        log_print(f"错误：权重文件不存在！{BEST_MODEL_PATH}")
        return

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    log_print("V8 最佳权重加载成功！\n")

    # ====== 1. 干净测试集评估（无RandomErasing）======
    log_print("=" * 60)
    log_print("【评估1：干净测试集（修复后）】")
    log_print("=" * 60)

    tc, tt = 0, 0
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
    normal_recall = tcc[0] / max(tct[0], 1)
    pneumonia_recall = tcc[1] / max(tct[1], 1)
    balanced_acc = 0.5 * normal_recall + 0.5 * pneumonia_recall

    log_print(f"测试集总准确率: {test_acc*100:.2f}%")
    log_print(f"  NORMAL: {tcc[0]}/{tct[0]} ({normal_recall*100:.1f}%)")
    log_print(f"  PNEUMONIA: {tcc[1]}/{tct[1]} ({pneumonia_recall*100:.1f}%)")
    log_print(f"  平衡准确率: {balanced_acc*100:.2f}%\n")

    # ====== 2. TTA 测试时增强（修复后：无RandomErasing）======
    log_print("=" * 60)
    log_print(f"【评估2：TTA测试时增强 (TTA_NUM={TTA_NUM})】")
    log_print("=" * 60)

    tc_tta = tt_tta = 0
    tta_correct_by_class = [0, 0]
    tta_total_by_class = [0, 0]

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            img_path, lbl = test_dataset.samples[idx]
            pil_img = Image.open(img_path).convert('L')

            # 原始预测
            base_img = test_transform(pil_img).unsqueeze(0).to(device)
            base_out = model(base_img) / TEMPERATURE
            votes = torch.softmax(base_out, dim=1)

            # TTA 增强预测（干净的tta_transform）
            for _ in range(TTA_NUM):
                aug_img = tta_transform(pil_img).unsqueeze(0).to(device)
                aug_out = model(aug_img) / TEMPERATURE
                votes += torch.softmax(aug_out, dim=1)

            avg_probs = votes / (TTA_NUM + 1)
            _, pred = torch.max(avg_probs, 1)
            tt_tta += 1
            tta_total_by_class[lbl] += 1
            if pred.item() == lbl:
                tc_tta += 1
                tta_correct_by_class[lbl] += 1

    tta_acc = tc_tta / tt_tta
    log_print(f"TTA测试集准确率: {tta_acc*100:.2f}%")
    log_print(f"  NORMAL TTA: {tta_correct_by_class[0]}/{tta_total_by_class[0]} ({tta_correct_by_class[0]/max(tta_total_by_class[0],1)*100:.1f}%)")
    log_print(f"  PNEUMONIA TTA: {tta_correct_by_class[1]}/{tta_total_by_class[1]} ({tta_correct_by_class[1]/max(tta_total_by_class[1],1)*100:.1f}%)\n")

    # ====== 3. 与原始报告的对比 ======
    log_print("=" * 60)
    log_print("【对比：原始报告 vs 修复后真实基线】")
    log_print("=" * 60)
    log_print(f"{'指标':<25} {'原始报告(有Bug)':<20} {'修复后(真实)':<20}")
    log_print(f"{'测试集 Acc':<25} {'87.02%':<20} {test_acc*100:.2f}%")
    log_print(f"{'NORMAL 召回率':<25} {'68.8%':<20} {normal_recall*100:.1f}%")
    log_print(f"{'PNEUMONIA 召回率':<25} {'97.9%':<20} {pneumonia_recall*100:.1f}%")
    log_print(f"{'平衡测试集 Acc':<25} {'83.38%':<20} {balanced_acc*100:.2f}%")
    log_print(f"{'TTA 测试集 Acc':<25} {'86.86%':<20} {tta_acc*100:.2f}%")

    # ====== 最终评分 ======
    log_print(f"\n{'='*60}")
    log_print("【修复后最终评分】")
    log_print(f"{'='*60}")

    # V8 光占比 Ro（复用V8报告中的计算值）
    # 光计算层参数
    optical_ops = 144 + 4608 + 18432 + 73728 + 64 + 128 + 256 + 512 + 2097152 + 16384 + 128
    electrical_ops = 16*64*64 + 32*32*32 + 64*16*16 + 128*8*8 + 384 + 64
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops if total_ops > 0 else 0

    log_print(f"光占比 Ro: {Ro*100:.2f}%")
    log_print(f"修复后测试集 Acc: {test_acc*100:.2f}%")
    log_print(f"修复后TTA Acc: {tta_acc*100:.2f}%")

    G = 1 if (Ro > 0.5 and test_acc > 0.85) else 0
    S_ratio = 20 * (Ro - 0.5) / 0.5
    S_acc = 5 * ((max(test_acc, 0.85) - 0.85) / 0.15) ** 2

    # 如果G=0（未过85%门槛），使用TTA Acc再算一次
    G_tta = 1 if (Ro > 0.5 and tta_acc > 0.85) else 0
    S_acc_tta = 5 * ((max(tta_acc, 0.85) - 0.85) / 0.15) ** 2

    log_print(f"\n使用测试集Acc评分:")
    log_print(f"  G={'通过' if G==1 else '未通过'} | S_ratio={S_ratio:.2f}/20 | S_acc={S_acc:.2f}/5")
    log_print(f"  总分: {G * (S_ratio + S_acc):.2f}/30")

    log_print(f"\n使用TTA Acc评分:")
    log_print(f"  G={'通过' if G_tta==1 else '未通过'} | S_ratio={S_ratio:.2f}/20 | S_acc_tta={S_acc_tta:.2f}/5")
    log_print(f"  总分(TTA): {G_tta * (S_ratio + S_acc_tta):.2f}/30")

    log_print(f"\n===== V8 真实基线评估完成 =====")


if __name__ == "__main__":
    main()
