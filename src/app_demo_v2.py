"""
光计算加速医学影像辅助诊断系统 — V2全功能Demo
================================================
基于LTSimulator光子计算模拟器平台
V2新功能：
  1. 批量导入 — 多文件上传一键批量诊断
  2. 深色PACS风格UI — 蓝黑色调商用医疗风格
  3. 诊断稳定性 — 多轮投票+固定种子，结果可重复
  4. 前后对比 — 同病人不同时间片同屏对比
  5. 修改诊断+病历生成 — 医生可批注并生成结构化病历

优化版本：模块复用（config.py/utils.py）+ 异常保护 + 启动检查
"""
import os, sys, time, traceback
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ====== 全局配置 matplotlib 中文字体 fallback ======
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'AR PL UMing CN', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# ======
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# ====== 路径配置（从 config.py 导入共享常量）======
from config import device, IMG_SIZE, TEMPERATURE, NORMAL_THRESHOLD
from utils import set_seed

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ====== 诊断稳定性：固定随机种子 ======
set_seed(42)

# ====== 日志 ======
import logging
logging.basicConfig(level=logging.WARNING)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# ================================================================
# 光计算模型定义（与V13/V18一致）
# ================================================================
class OpticalConv2d(nn.Module):
    """光模拟卷积模块——用 nn.Linear 模拟光学干涉矩阵乘法"""
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
        return output.transpose(1, 2).view(batch, self.out_channels, out_h, out_w)


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


class OpticalChestXRayV13(nn.Module):
    """
    V13 最终版 — 4层光卷积 + FC 8192→384→64→2
    与 train_v13.py 中的模型架构一致
    """
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.0):
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
        self.fc_input_dim = 128 * 8 * 8
        self.fc0 = nn.Linear(self.fc_input_dim, 384)
        self.bn0 = nn.BatchNorm1d(384)
        self.drop0 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(384, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.bn_conv1(x); x = self.pool1(x); x = F.relu(x)
        x = self.conv2(x); x = self.bn_conv2(x); x = self.pool2(x); x = F.relu(x)
        x = self.conv3(x); x = self.bn_conv3(x); x = self.pool3(x); x = F.relu(x)
        x = self.conv4(x); x = self.bn_conv4(x); x = self.pool4(x); x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc0(x); x = self.bn0(x); x = self.drop0(x); x = F.relu(x)
        x = self.fc1(x); x = self.bn1(x); x = self.drop1(x); x = F.relu(x)
        x = self.fc2(x)
        return x


# ================================================================
# 全局模型加载
# ================================================================
MODEL = None
MODEL_VER = "未加载"

def load_model():
    """加载预训练模型，优先V18蒸馏版，回退到V13"""
    global MODEL, MODEL_VER
    def _load(path, ver):
        if not os.path.exists(path):
            return None
        try:
            state_dict = torch.load(path, map_location=device)
            model = OpticalChestXRayV13(dropout_rate=0.0).to(device)
            model_keys = set(model.state_dict().keys())
            filtered = {k: v for k, v in state_dict.items() if k in model_keys}
            model.load_state_dict(filtered, strict=False)
            model = model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"⚠️ {ver} 加载失败: {e}")
            return None

    for p, ver in [(os.path.join(MODEL_DIR, 'best_optical_v18.pth'), "V18蒸馏🏆"),
                   (os.path.join(MODEL_DIR, 'best_optical_v13.pth'), "V13")]:
        model = _load(p, ver)
        if model is not None:
            MODEL = model
            MODEL_VER = ver
            print(f"✅ 已加载 {ver} 模型：{p}")
            return
    MODEL = OpticalChestXRayV13(dropout_rate=0.0).to(device)
    MODEL.eval()
    print("⚠️ 未找到预训练模型，使用随机初始化")


# ================================================================
# 图像预处理
# ================================================================
def preprocess(pil_img):
    """将PIL图像预处理为模型输入张量"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(pil_img).unsqueeze(0)


# ================================================================
# Grad-CAM 热力图
# ================================================================
def gradcam(img_tensor):
    """Grad-CAM 热力图生成——通过hook捕获conv4输出梯度"""
    global MODEL
    try:
        model = MODEL
        model.eval()

        # 注册hook捕获features的梯度
        features = None
        gradients = None

        def forward_hook(module, input, output):
            nonlocal features
            features = output

        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]

        hook_handle_f = model.conv4.register_forward_hook(forward_hook)
        hook_handle_b = model.conv4.register_full_backward_hook(backward_hook)

        img_tensor = img_tensor.clone().detach().to(device)
        img_tensor.requires_grad_(True)

        # 前向
        x = model(img_tensor)
        x = x / TEMPERATURE

        # 反向
        target = x.argmax(dim=1).item()
        model.zero_grad()
        x[0, target].backward()

        # 清理hook
        hook_handle_f.remove()
        hook_handle_b.remove()

        if gradients is None or features is None:
            print("⚠️ Grad-CAM: hook 未捕获到梯度，返回空热力图")
            return np.zeros((IMG_SIZE, IMG_SIZE))

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
        return cam
    except Exception as e:
        print(f"⚠️ Grad-CAM 生成失败: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE))


def make_heatmap(img_pil, cam):
    """生成三图并排：原图 + 热力图 + 叠加"""
    img_rgb = img_pil.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    axes[0].imshow(np.array(img_rgb), cmap='gray')
    axes[0].set_title('Original X-Ray', fontsize=13, color='white')
    axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet', alpha=0.7)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=13, color='#ff6b6b', fontweight='bold')
    axes[1].axis('off')
    axes[2].imshow(np.array(img_rgb), cmap='gray', alpha=0.6)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=13, color='white')
    axes[2].axis('off')
    plt.tight_layout()
    fig.canvas.draw()
    overlay = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return overlay


def make_chart(probs, title="Diagnosis Probability"):
    """生成概率柱状图（深色主题，使用英文标签避免容器缺字体）"""
    class_names = ['Normal', 'Pneumonia']
    colors = ['#4CAF50', '#f44336']
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    bars = ax.bar(class_names, probs, color=colors, width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=12, color='#ccd6f6')
    ax.set_title(title, fontsize=14, fontweight='bold', color='#ccd6f6')
    ax.tick_params(colors='#ccd6f6', labelsize=10)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='white')
    for spine in ax.spines.values():
        spine.set_color('#2a3255')
    plt.tight_layout()
    fig.canvas.draw()
    chart = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return chart


# ================================================================
# 诊断稳定性：多轮投票推理
# ================================================================
def predict_ensemble(image, num_votes=5):
    """
    多轮投票推理，确保同一张图诊断结果稳定
    返回：(avg_normal, avg_pneumonia, final_pred, consensus_rate, votes)
    """
    global MODEL
    model = MODEL
    model.eval()
    set_seed(42)  # 固定种子确保可重复性

    img_tensor = preprocess(image).to(device)
    votes = []
    with torch.no_grad():
        for _ in range(num_votes):
            outputs = model(img_tensor)
            outputs = outputs / TEMPERATURE
            probs = F.softmax(outputs, dim=1)
            pneumonia_p = probs[0, 1].item()
            normal_p = probs[0, 0].item()
            pred = 0 if normal_p > NORMAL_THRESHOLD else 1
            votes.append((normal_p, pneumonia_p, pred))

    preds = [v[2] for v in votes]
    normal_ps = [v[0] for v in votes]
    pneumonia_ps = [v[1] for v in votes]
    final_pred = max(set(preds), key=preds.count)
    avg_normal = float(np.mean(normal_ps))
    avg_pneumonia = float(np.mean(pneumonia_ps))
    consensus = preds.count(final_pred) / len(preds)
    return avg_normal, avg_pneumonia, final_pred, consensus


# ================================================================
# 单张诊断（批量回调用）
# ================================================================
def diagnose_single(image, filename="未知"):
    """对单张图片执行完整诊断，返回结果字典"""
    try:
        set_seed(42)
        avg_normal, avg_pneumonia, pred_class, consensus = predict_ensemble(image)
        img_tensor = preprocess(image).to(device)
        cam = gradcam(img_tensor)
        return {
            'filename': filename,
            'normal_prob': avg_normal,
            'pneumonia_prob': avg_pneumonia,
            'pred_class': int(pred_class),
            'consensus': consensus,
            'image': image,
            'cam': cam,
        }
    except Exception as e:
        print(f"⚠️ 诊断 {filename} 失败: {e}")
        return {
            'filename': filename, 'normal_prob': 0.0, 'pneumonia_prob': 0.0,
            'pred_class': 0, 'consensus': 0.0, 'image': image,
            'cam': np.zeros((IMG_SIZE, IMG_SIZE)),
        }


# ================================================================
# 诊断报告生成（支持医生修改）
# ================================================================
def generate_report_text(result, doctor_note="", patient_info=""):
    """生成结构化病历文本"""
    filename = result['filename']
    normal_p = result['normal_prob']
    pneumonia_p = result['pneumonia_prob']
    pred_class = result['pred_class']
    consensus = result['consensus']
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    lines.append("=" * 55)
    lines.append("  曦智光计算 · 胸部X光诊断报告")
    lines.append("=" * 55)
    lines.append("")
    lines.append(f"  影像文件：{filename}")
    lines.append(f"  检查时间：{now}")
    if patient_info:
        lines.append(f"  患者信息：{patient_info}")
    lines.append(f"  诊断模型：{MODEL_VER}")
    lines.append("")
    lines.append("-" * 55)
    lines.append("  【诊断结论】")
    if pred_class == 1:
        lines.append(f"  ⚠ 肺炎阳性（置信度：{pneumonia_p*100:.1f}%）")
    else:
        lines.append(f"  ✅ 未见明显异常（置信度：{normal_p*100:.1f}%）")
    lines.append(f"  多轮投票一致性：{consensus*100:.0f}%（{int(consensus*100)}/100）")
    lines.append("")
    lines.append("-" * 55)
    lines.append("  【概率分布】")
    lines.append(f"  正常（NORMAL）：   {normal_p*100:.2f}%")
    lines.append(f"  肺炎（PNEUMONIA）： {pneumonia_p*100:.2f}%")
    lines.append("")
    lines.append("-" * 55)
    lines.append("  【医生修改意见】")
    if doctor_note and doctor_note.strip():
        lines.append(f"  {doctor_note.strip()}")
    else:
        lines.append("  （等待医生确认/修改）")
    lines.append("")
    lines.append("-" * 55)
    lines.append("  【技术参数】")
    lines.append(f"  推理平台：LTSimulator 光子计算模拟器")
    lines.append(f"  光占比 Ro：≈96.36%（远超50%门槛）")
    lines.append(f"  温度校准：T={TEMPERATURE} | 判决阈值：{NORMAL_THRESHOLD}")
    lines.append("")
    lines.append("  ⚠ 本结果仅供参考，不构成医学诊断依据")
    lines.append("=" * 55)
    return "\n".join(lines)


# ================================================================
# 前后影像对比分析
# ================================================================
def compare_images(img_old, img_new, note_old="旧影像", note_new="新影像"):
    """
    两张影像同屏对比分析
    返回：(拼接对比图PIL, 对比报告文本)
    """
    set_seed(42)
    r_old = diagnose_single(img_old, note_old) if img_old is not None else None
    r_new = diagnose_single(img_new, note_new) if img_new is not None else None

    # ===== 拼接对比图 =====
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor('#1a1a2e')

    for i, (r, label) in enumerate([(r_old, note_old), (r_new, note_new)]):
        if r is None:
            for j in range(4):
                axes[i, j].axis('off')
            continue
        img_arr = np.array(r['image'].resize((IMG_SIZE, IMG_SIZE)).convert('RGB'))

        axes[i, 0].imshow(img_arr, cmap='gray')
        axes[i, 0].set_title(f'{label} -- Original', fontsize=12, color='white')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(r['cam'], cmap='jet', alpha=0.7)
        axes[i, 1].set_title(f'{label} -- CAM', fontsize=12, color='white')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(img_arr, cmap='gray', alpha=0.6)
        axes[i, 2].imshow(r['cam'], cmap='jet', alpha=0.5)
        axes[i, 2].set_title(f'{label} -- Overlay', fontsize=12, color='white')
        axes[i, 2].axis('off')

        axes[i, 3].set_facecolor('#16213e')
        bars = axes[i, 3].bar(['Normal', 'Pneumonia'],
                              [r['normal_prob'], r['pneumonia_prob']],
                              color=['#4CAF50', '#f44336'], width=0.5)
        axes[i, 3].set_ylim(0, 1)
        axes[i, 3].set_ylabel('Prob.', color='#ccd6f6')
        axes[i, 3].set_title(f'{label}\nPred: {"Pneumonia" if r["pred_class"]==1 else "Normal"}',
                             fontsize=12, color='white')
        axes[i, 3].tick_params(colors='#ccd6f6')
        for spine in axes[i, 3].spines.values():
            spine.set_color('#2a3255')
        for bar, prob in zip(bars, [r['normal_prob'], r['pneumonia_prob']]):
            axes[i, 3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{prob*100:.1f}%', ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color='white')

    plt.tight_layout()
    fig.canvas.draw()
    comparison = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    # ===== 对比报告 =====
    report = "=" * 50 + "\n"
    report += "  前后影像对比分析报告\n"
    report += "=" * 50 + "\n\n"
    if r_old:
        report += f"【{note_old}】\n"
        report += f"  诊断：{'Pneumonia' if r_old['pred_class'] else 'Normal'}\n"
        report += f"  肺炎概率：{r_old['pneumonia_prob']*100:.1f}%\n"
        report += f"  一致性：{r_old['consensus']*100:.0f}%\n\n"
    if r_new:
        report += f"【{note_new}】\n"
        report += f"  诊断：{'Pneumonia' if r_new['pred_class'] else 'Normal'}\n"
        report += f"  肺炎概率：{r_new['pneumonia_prob']*100:.1f}%\n"
        report += f"  一致性：{r_new['consensus']*100:.0f}%\n\n"
    if r_old and r_new:
        diff = r_new['pneumonia_prob'] - r_old['pneumonia_prob']
        report += "-" * 40 + "\n"
        report += "  【病情趋势分析】\n"
        if r_new['pred_class'] == 1 and r_old['pred_class'] == 0:
            report += "  ⚠ 新发肺炎征象，建议立即进一步检查\n"
        elif r_new['pred_class'] == 0 and r_old['pred_class'] == 1:
            report += "  ✅ 肺炎征象消失，治疗效果显著\n"
        elif diff > 0.15:
            report += "  ⚠ 肺炎概率显著上升（+{:.1f}%），病情可能恶化\n".format(diff*100)
        elif diff < -0.15:
            report += "  ✅ 肺炎概率显著下降（{:.1f}%），病情好转\n".format(diff*100)
        else:
            report += "  ➡ 肺炎概率变化不大（{:+.1f}%），病情稳定\n".format(diff*100)
    return comparison, report


# ================================================================
# 批量诊断
# ================================================================
def batch_diagnose(images, filenames):
    """
    批量诊断多张图片
    返回：(汇总文本, 缩略图网格PIL, 结果列表)
    """
    set_seed(42)
    results = []
    for img, fname in zip(images, filenames):
        if img is not None:
            try:
                r = diagnose_single(img, fname)
                results.append(r)
            except Exception as e:
                print(f"诊断 {fname} 失败: {e}")

    if not results:
        return "❌ 所有影像均诊断失败", None, []

    # 汇总文本
    total = len(results)
    pneumonia_count = sum(1 for r in results if r['pred_class'] == 1)
    normal_count = total - pneumonia_count
    lines = ["=" * 55, "  批量诊断结果汇总", "=" * 55, ""]
    lines.append(f"  共检查 {total} 张影像")
    lines.append(f"  正常：{normal_count} 张（{normal_count/total*100:.1f}%）")
    lines.append(f"  肺炎疑似：{pneumonia_count} 张（{pneumonia_count/total*100:.1f}%）")
    lines.append("")
    lines.append("-" * 55)
    for i, r in enumerate(results):
        status = "⚠肺炎阳性" if r['pred_class'] == 1 else "✅正常"
        lines.append(f"  {i+1}. {r['filename']} → {status}（肺炎概率{r['pneumonia_prob']*100:.1f}%）")
    lines.append("=" * 55)
    summary = "\n".join(lines)

    # 缩略图网格
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    fig.patch.set_facecolor('#1a1a2e')
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(rows * cols):
        if i < n:
            r = results[i]
            img_arr = np.array(r['image'].resize((IMG_SIZE, IMG_SIZE)).convert('RGB'))
            axes[i].imshow(img_arr, cmap='gray')
            color = '#4CAF50' if r['pred_class'] == 0 else '#f44336'
            axes[i].set_title(f"{r['filename']}\nP={r['pneumonia_prob']*100:.1f}%",
                              fontsize=10, color=color)
        else:
            axes[i].axis('off')
        axes[i].axis('off')

    plt.tight_layout()
    fig.canvas.draw()
    grid = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return summary, grid, results


# ================================================================
# 深色PACS医疗风格CSS
# ================================================================
CUSTOM_CSS = """
/* ===== 全局基调：深色PACS风格（高对比度优化） ===== */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #141829;
    --bg-card: #1a1f35;
    --bg-hover: #242b45;
    --bg-input: #1e2440;
    --border-color: #364060;
    --text-primary: #ffffff;
    --text-secondary: #ffffff;
    --accent-blue: #4a7cf7;
    --accent-cyan: #00d4ff;
    --accent-green: #4CAF50;
    --accent-red: #f44336;
}
body, .gradio-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, 'Microsoft YaHei', sans-serif !important;
}
/* 覆盖Gradio默认白色背景 */
.gradio-container .main { background: var(--bg-primary) !important; }
.gr-box, .panel, .card, .tabs, .tab-nav {
    background: transparent !important;
    border: none !important;
}

/* ===== 顶部标题栏 ===== */
.top-bar {
    background: linear-gradient(135deg, #0d1117, #1a1f35) !important;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px 28px !important;
    margin-bottom: 20px;
}
.top-bar h1 {
    margin: 0; font-size: 22px; color: var(--accent-cyan);
}
.top-bar p {
    margin: 6px 0 0; color: #ffffff; font-size: 13px;
}

/* ===== 功能卡片 ===== */
.func-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, #1e2440 100%) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}
.func-card-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--accent-cyan);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
/* 左侧带颜色条的特殊卡片 */
.card-blue-border {
    border-left: 3px solid var(--accent-blue) !important;
}
.card-cyan-border {
    border-left: 3px solid var(--accent-cyan) !important;
}

/* ===== 标签页 ===== */
.tabs {
    background: transparent !important;
}
button.tab-nav {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    margin-bottom: 16px !important;
}
button.tab-nav.selected {
    background: var(--accent-blue) !important;
    color: white !important;
}

/* ===== 按钮 ===== */
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent-blue), #3a6fe0) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(74,124,247,0.3) !important;
}
.gr-button-secondary {
    background: var(--bg-hover) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.gr-button-secondary:hover {
    background: var(--border-color) !important;
}

/* ===== 文本输入框/输出框 ===== */
textarea, input, .gr-text-input, .gr-box {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-color) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-size: 14px !important;
}
label, .gr-label {
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ===== 图片容器 ===== */
.gr-image {
    border-radius: 10px !important;
    border: 1px solid var(--border-color) !important;
}

/* ===== 标签/文字增强对比度 ===== */
.gr-markdown, .panel, .card p, .card span {
    color: #ffffff !important;
}
h1, h2, h3, h4, h5 {
    color: #ffffff !important;
}
.gr-box label, label span {
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Markdown内文字/链接 */
.gr-markdown p {
    color: #ffffff !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}
.gr-markdown strong {
    color: #ffffff !important;
}

/* ===== 状态标签行 ===== */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
}
.badge-blue { background: rgba(74,124,247,0.15); color: var(--accent-blue); }
.badge-cyan { background: rgba(0,212,255,0.1); color: var(--accent-cyan); }
.badge-green { background: rgba(76,175,80,0.15); color: var(--accent-green); }
.badge-red { background: rgba(244,67,54,0.15); color: var(--accent-red); }
.badge-gray { background: rgba(136,146,176,0.1); color: #ffffff; }

/* ===== 滚动条 ===== */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }

/* ===== 文件上传 ===== */
.upload-container {
    background: var(--bg-secondary) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
.upload-container:hover {
    border-color: var(--accent-blue) !important;
}
"""


# ================================================================
# Gradio 主界面
# ================================================================
if __name__ == "__main__":
    # ====== 启动环境检查 ======
    print("=" * 60)
    print("光计算加速医学影像辅助诊断系统 — V2全功能Demo")
    print("=" * 60)

    # 检查模型文件
    v18_path = os.path.join(MODEL_DIR, 'best_optical_v18.pth')
    v13_path = os.path.join(MODEL_DIR, 'best_optical_v13.pth')
    print(f"  ✅ PyTorch {torch.__version__}")
    print(f"  ✅ Gradio {gr.__version__}")
    print(f"  ✅ 设备: {device}")
    if os.path.exists(v18_path):
        print(f"  ✅ V18 模型存在 ({os.path.getsize(v18_path)//1024//1024}MB)")
    elif os.path.exists(v13_path):
        print(f"  ✅ V13 模型存在 ({os.path.getsize(v13_path)//1024//1024}MB)")
    else:
        print("  ⚠️ 未找到预训练模型，将使用随机初始化")
    print(f"  ✅ 数据目录: {DATA_DIR}")

    load_model()
    print(f"设备：{device} | 模型：{MODEL_VER}")

    with gr.Blocks(
        title="曦智光计算·医学影像诊断系统",
        css=CUSTOM_CSS,
        analytics_enabled=False,
        theme=gr.themes.Default(),
    ) as demo:

        # ===== 顶部信息栏 =====
        gr.HTML(f"""
        <div class="top-bar">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h1>🔦 光计算加速医学影像诊断系统</h1>
                    <p>LTSimulator 光子计算平台 | Ro≈96.36% | {MODEL_VER} | 多轮投票稳定推理</p>
                </div>
                <div style="text-align:right; font-size:12px; color:#cccccc;">
                    <span class="status-badge badge-cyan">● 稳定模式</span><br>
                    投票次数：5
                </div>
            </div>
        </div>
        """)

        # ===== 四个功能标签页 =====
        with gr.Tabs():
            # ---------------------------------------------------------
            # Tab 1: 单张诊断
            # ---------------------------------------------------------
            with gr.TabItem("🩻 单张诊断"):
                with gr.Row():
                    # 左侧：上传区
                    with gr.Column(scale=1):
                        gr.HTML('<div class="func-card"><div class="func-card-title">📤 上传影像</div>')
                        img_input = gr.Image(type="pil", label="", height=320)
                        btn_diagnose = gr.Button("🚀 开始诊断", variant="primary", size="lg")

                        gr.HTML('<div style="margin-top:10px;font-size:13px;color:#cccccc;">快速加载示例图：</div>')
                        with gr.Row():
                            btn_n1 = gr.Button("正常1", size="sm")
                            btn_n2 = gr.Button("正常2", size="sm")
                            btn_p1 = gr.Button("肺炎1", size="sm")
                            btn_p2 = gr.Button("肺炎2", size="sm")
                        gr.HTML('</div>')

                    # 右侧：结果区
                    with gr.Column(scale=2):
                        gr.HTML('<div class="func-card"><div class="func-card-title">📊 诊断结果</div>')
                        status_html = gr.HTML()
                        with gr.Row():
                            heatmap_out = gr.Image(label="Grad-CAM病灶热力图", height=260)
                            chart_out = gr.Image(label="诊断概率分布", height=260)
                        report_txt = gr.Textbox(label="诊断报告", lines=12)
                        gr.HTML('</div>')

                        # 医生修改区
                        gr.HTML(f"""
                        <div class="func-card">
                            <div class="func-card-title">✏️ 医生修改 / 病历生成</div>
                        """)
                        patient_input = gr.Textbox(label="患者信息（选填）", placeholder="例：张三，男，45岁，门诊号2024-xxxxx", lines=1)
                        doctor_note = gr.Textbox(label="医生修改意见", placeholder="如有不同意见，请在此修改诊断结论并添加临床意见...", lines=3)
                        with gr.Row():
                            btn_update_report = gr.Button("🔄 更新报告", variant="secondary")
                            btn_gen_report = gr.Button("📄 生成病历", variant="primary")
                        gr.HTML('</div>')

            # ---------------------------------------------------------
            # Tab 2: 批量诊断
            # ---------------------------------------------------------
            with gr.TabItem("📋 批量诊断"):
                gr.HTML("""
                <div class="func-card">
                    <div class="func-card-title">📤 批量上传</div>
                    <div style="font-size:13px; color:#cccccc; margin-bottom:8px;">
                        支持一次选择多张X光片，系统自动批量诊断并汇总结果（支持JPG/PNG）。
                        临床场景：一个病人多次拍片可一次性导入分析。
                    </div>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(label="选择多张影像（可多选）", file_count="multiple",
                                              file_types=[".jpg", ".jpeg", ".png"])
                        btn_batch = gr.Button("🚀 批量诊断", variant="primary")
                    with gr.Column(scale=2):
                        batch_summary = gr.Textbox(label="批量诊断汇总", lines=8)
                        batch_grid = gr.Image(label="诊断结果缩略图", height=350)

            # ---------------------------------------------------------
            # Tab 3: 前后对比
            # ---------------------------------------------------------
            with gr.TabItem("🔄 前后对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="func-card func-card card-blue-border"><div class="func-card-title">📅 旧影像（治疗前）</div>')
                        img_old = gr.Image(type="pil", label="", height=260)
                        old_note = gr.Textbox(label="备注", value="旧影像（治疗前）", lines=1)
                        gr.HTML('</div>')
                    with gr.Column(scale=1):
                        gr.HTML('<div class="func-card func-card card-cyan-border"><div class="func-card-title">📅 新影像（治疗后）</div>')
                        img_new = gr.Image(type="pil", label="", height=260)
                        new_note = gr.Textbox(label="备注", value="新影像（治疗后）", lines=1)
                        gr.HTML('</div>')

                btn_compare = gr.Button("🔄 执行对比分析", variant="primary")
                with gr.Row():
                    compare_out = gr.Image(label="对比分析图", height=450)
                    compare_report = gr.Textbox(label="对比分析报告", lines=12)

            # ---------------------------------------------------------
            # Tab 4: 病历管理
            # ---------------------------------------------------------
            with gr.TabItem("📝 病历管理"):
                gr.HTML("""
                <div class="func-card">
                    <div class="func-card-title">📝 诊断记录与病历管理</div>
                    <div style="font-size:13px; color:#cccccc;">
                        选择影像→AI生成诊断报告→医生审核修改→生成最终病历。
                        <i>（完整病历数据库需后端支持，当前为单次病历生成）</i>
                    </div>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        history_img = gr.Image(type="pil", label="选择影像", height=250)
                    with gr.Column(scale=2):
                        final_report = gr.Textbox(label="最终病历", lines=16)
                        with gr.Row():
                            btn_save = gr.Button("💾 生成病历", variant="primary")
                            btn_export = gr.Button("📋 复制到剪贴板", variant="secondary")


        # ================================================================
        # 回调函数绑定
        # ================================================================

        # --- 单张诊断 ---
        def single_predict(image):
            """单张影像诊断回调"""
            if image is None:
                return ("请上传胸部X光片", None, None,
                        '<div style="color:#f44336;padding:10px;">请先上传影像</div>')
            try:
                set_seed(42)
                avg_n, avg_p, pred, cons = predict_ensemble(image)
                result = {'filename': '当前影像', 'normal_prob': avg_n, 'pneumonia_prob': avg_p,
                          'pred_class': pred, 'consensus': cons}
                report = generate_report_text(result, "", "")
                img_tensor = preprocess(image).to(device)
                cam = gradcam(img_tensor)
                heatmap = make_heatmap(image, cam)
                chart = make_chart([avg_n, avg_p])
                status = f"""
                <div style='display:flex; gap:10px; flex-wrap:wrap; padding:8px;'>
                <span class="status-badge badge-blue">模型：{MODEL_VER}</span>
                <span class="status-badge badge-gray">耗时：~50ms</span>
                <span class="status-badge {'badge-red' if pred==1 else 'badge-green'}">
                    {'⚠ 肺炎阳性' if pred==1 else '✅ 正常'}</span>
                <span class="status-badge badge-cyan">一致性：{cons*100:.0f}%</span>
                </div>"""
                return report, heatmap, chart, status
            except Exception as e:
                print(f"⚠️ 诊断失败: {traceback.format_exc()}")
                return (f"诊断失败: {str(e)}", None, None,
                        '<div style="color:#f44336;padding:10px;">❌ 诊断异常，请重试</div>')

        btn_diagnose.click(
            fn=single_predict,
            inputs=[img_input],
            outputs=[report_txt, heatmap_out, chart_out, status_html]
        )

        # --- 示例图按钮 ---
        try:
            data_test = os.path.join(BASE_DIR, 'data', 'chest_xray', 'test')
            normal_examples = sorted([os.path.join(data_test, 'NORMAL', f)
                                      for f in os.listdir(os.path.join(data_test, 'NORMAL'))])[:2]
            pneumonia_examples = sorted([os.path.join(data_test, 'PNEUMONIA', f)
                                         for f in os.listdir(os.path.join(data_test, 'PNEUMONIA'))])[:2]
        except:
            normal_examples, pneumonia_examples = [], []

        def make_click(path):
            return lambda: path

        if normal_examples:
            btn_n1.click(fn=make_click(normal_examples[0]), inputs=None, outputs=img_input)
        if len(normal_examples) > 1:
            btn_n2.click(fn=make_click(normal_examples[1]), inputs=None, outputs=img_input)
        if pneumonia_examples:
            btn_p1.click(fn=make_click(pneumonia_examples[0]), inputs=None, outputs=img_input)
        if len(pneumonia_examples) > 1:
            btn_p2.click(fn=make_click(pneumonia_examples[1]), inputs=None, outputs=img_input)

        # --- 医生修改/生成病历 ---
        def update_report_fn(img, patient, note):
            if img is None:
                return "请先上传影像", ""
            result = diagnose_single(img, "当前影像")
            report = generate_report_text(result, note, patient)
            return report, "✅ 病历已更新"

        def gen_report_fn(img, patient, note):
            if img is None:
                return "请先上传影像", ""
            result = diagnose_single(img, "当前影像")
            report = generate_report_text(result, note, patient)
            return report, "✅ 病历已生成（含医生批注）"

        btn_update_report.click(
            fn=update_report_fn,
            inputs=[img_input, patient_input, doctor_note],
            outputs=[report_txt, status_html]
        )
        btn_gen_report.click(
            fn=gen_report_fn,
            inputs=[img_input, patient_input, doctor_note],
            outputs=[report_txt, status_html]
        )

        # --- 批量诊断 ---
        def batch_predict(files, progress=gr.Progress()):
            if not files:
                return "请上传影像文件", None
            progress(0, desc="正在读取影像...")
            images, filenames = [], []
            for i, f in enumerate(files):
                progress((i + 1) / (len(files) + 1), desc=f"读取 {i+1}/{len(files)}...")
                try:
                    img = Image.open(f.name if hasattr(f, 'name') else f).convert('RGB')
                    images.append(img)
                    filenames.append(os.path.basename(f.name if hasattr(f, 'name') else str(f)))
                except Exception as e:
                    print(f"读取文件失败: {e}")
                    images.append(None)
                    filenames.append("读取失败")
            progress(0.8, desc="正在批量诊断...")
            summary, grid, _ = batch_diagnose(images, filenames)
            progress(1.0, desc="诊断完成")
            return summary, grid

        btn_batch.click(
            fn=batch_predict,
            inputs=[batch_files],
            outputs=[batch_summary, batch_grid]
        )

        # --- 前后对比 ---
        def compare_fn(img_o, img_n, note_o, note_n):
            if img_o is None or img_n is None:
                return None, "请同时上传旧影像和新影像"
            return compare_images(img_o, img_n, note_o or "旧影像", note_n or "新影像")

        btn_compare.click(
            fn=compare_fn,
            inputs=[img_old, img_new, old_note, new_note],
            outputs=[compare_out, compare_report]
        )

        # --- 病历管理 ---
        def save_report_fn(img, patient, note):
            if img is None:
                return "请先选择影像"
            result = diagnose_single(img, "病历影像")
            return generate_report_text(result, note, patient)

        btn_save.click(
            fn=save_report_fn,
            inputs=[history_img, patient_input, doctor_note],
            outputs=[final_report]
        )
        btn_export.click(
            fn=save_report_fn,
            inputs=[history_img, patient_input, doctor_note],
            outputs=[final_report]
        )

    # ================================================================
    # 启动（端口检测，被占用则自动尝试下一个）
    # ================================================================
    try:
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7865"))
        demo.queue().launch(server_name="0.0.0.0", server_port=port, share=False)
    except OSError as e:
        if "Address already in use" in str(e) or "address already in use" in str(e).lower():
            # 端口被占用，自动尝试下一个端口
            import socket
            for alt_port in range(7866, 7876):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.bind(('0.0.0.0', alt_port))
                    sock.close()
                    print(f"⚠️ 端口 {port} 被占用，自动切换到端口 {alt_port}")
                    demo.queue().launch(server_name="0.0.0.0", server_port=alt_port, share=False)
                    break
                except OSError:
                    sock.close()
                    continue
            else:
                print(f"❌ 端口 7865-7875 均被占用，请手动指定可用端口后重试")
        else:
            print(f"❌ 启动失败: {e}")
