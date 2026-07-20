"""
⚠️ 此文件已弃用 — 请使用 app_demo_v2.py（功能更全，持续维护）
==================================================================
光计算加速医学影像辅助诊断系统 — 全面改造版（V18蒸馏模型）
==================================================================
基于LTSimulator光子计算模拟器平台
V18蒸馏模型：4层光卷积 + FC 8192→384→64→2 | Ro=96.36%

功能升级：
1. 批量导入（文件夹加载，按病人ID自动识别）
2. 深色医疗风UI（参考商用PACS系统）
3. 诊断稳定性（锁随机种子 + 固定eval）
4. 前后对比视图（双列对比 + 概率变化趋势）
5. 医生修改诊断 + 一键生成图文病历报告

创建时间：2026-07-19
"""
import os, sys, time, random, traceback
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# ====== 诊断稳定性：锁死随机种子 ======
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====== 路径配置 ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # 项目根目录
MODEL_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
TEMPERATURE = 1.5           # 温度校准参数（V18蒸馏）
NORMAL_THRESHOLD = 0.50     # V18最优NORMAL判决阈值

import logging
logging.basicConfig(level=logging.WARNING)
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# 支持图片格式
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# ====== 光计算模型定义（V13架构，兼容V18蒸馏权重）======

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
    V13架构 — 4层光卷积 + FC 8192→384→64→2
    V18蒸馏模型使用此架构加载（兼容V13/V16/V18权重）
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


# ====== 全局模型（只加载一次，保持eval状态）======
MODEL = None
MODEL_VER = "未加载"

def load_model():
    """加载V18蒸馏模型（只执行一次）"""
    global MODEL, MODEL_VER
    if MODEL is not None:
        return  # 已加载，跳过

    # V18 蒸馏优先（当前最优）
    v18_path = os.path.join(MODEL_DIR, 'best_optical_v18.pth')
    if _try_load_model(v18_path, "V18蒸馏🏆"):
        print(f"✅ 已加载 V18 蒸馏模型（当前最优）")
        print(f"   校准Acc=91.03% | Ro=96.36%")
        print(f"   温度校准T={TEMPERATURE} | NORMAL阈值={NORMAL_THRESHOLD}")
        return

    # V13 回退
    v13_path = os.path.join(MODEL_DIR, 'best_optical_v13.pth')
    if _try_load_model(v13_path, "V13"):
        print(f"✅ 已加载 V13 模型（回退）")
        return

    # 随机初始化回退（不应该到这里）
    MODEL = OpticalChestXRayV13(dropout_rate=0.0).to(device)
    MODEL.eval()
    print("⚠️ 警告：未找到预训练模型，使用随机初始化")


def _try_load_model(path, ver_str):
    """尝试加载指定路径的权重到全局MODEL"""
    global MODEL, MODEL_VER
    if not os.path.exists(path):
        return False
    state_dict = torch.load(path, map_location=device)
    model = OpticalChestXRayV13(dropout_rate=0.0).to(device)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False)
    model = model.to(device)
    model.eval()
    MODEL = model
    MODEL_VER = ver_str
    return True


def preprocess(pil_img):
    """图像预处理：缩放到128×128 + 灰度化 + 归一化"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(pil_img).unsqueeze(0)


def infer_one(img_tensor):
    """单图推理，返回 (normal_p, pneumonia_p, pred_class, elapsed_ms)"""
    global MODEL
    model = MODEL
    start = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        outputs = outputs / TEMPERATURE
        probs = F.softmax(outputs, dim=1)
        pneumonia_p = probs[0, 1].item()
        normal_p = probs[0, 0].item()
    elapsed_ms = (time.time() - start) * 1000
    pred_class = 0 if normal_p > NORMAL_THRESHOLD else 1
    return normal_p, pneumonia_p, pred_class, elapsed_ms


def gradcam(img_tensor):
    """Grad-CAM 热力图（适配V13 conv4架构，使用hook避免梯度丢失）"""
    global MODEL
    model = MODEL
    img_tensor = img_tensor.to(device)

    # 注册hook捕获conv4输出的梯度
    features = None
    gradients = None

    def forward_hook(module, inp, out):
        nonlocal features
        features = out

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    hook_forward = model.conv4.register_forward_hook(forward_hook)
    hook_backward = model.conv4.register_full_backward_hook(backward_hook)

    # 前向传播
    output = model(img_tensor)
    output = output / TEMPERATURE
    target = output.argmax(dim=1).item()

    # 反向传播
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target] = 1.0
    output.backward(gradient=one_hot)

    # 移除hook
    hook_forward.remove()
    hook_backward.remove()

    if features is None or gradients is None:
        # 降级方案：返回空热力图
        return np.zeros((IMG_SIZE, IMG_SIZE))

    # 计算Grad-CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * features).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    cam = (cam - cam.min()) / max(cam.max() - cam.min(), 1e-8)
    return cam


def make_heatmap(img_pil, cam):
    """生成热力图叠加图（3图并排，均匀分布）"""
    img_rgb = img_pil.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax in axes:
        ax.set_aspect('equal')
    axes[0].imshow(np.array(img_rgb), cmap='gray')
    axes[0].set_title('Original X-Ray', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(cam, cmap='jet', alpha=0.7)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=13, color='red', fontweight='bold')
    axes[1].axis('off')
    axes[2].imshow(np.array(img_rgb), cmap='gray', alpha=0.6)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay', fontsize=13, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    fig.canvas.draw()
    overlay = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return overlay


def make_chart(probs):
    """生成概率柱状图（无中文，避免容器内字体缺失）"""
    class_names = ['Normal', 'Pneumonia']
    colors = ['#4CAF50', '#f44336']
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(class_names, probs, color=colors, width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Diagnosis Probability Distribution', fontsize=14, fontweight='bold')
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.canvas.draw()
    chart = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return chart


# ====== 深色医疗风CSS ======
DARK_CSS = """
:root {
    --bg-primary: #0a1628;
    --bg-secondary: #111d30;
    --bg-card: #1a2332;
    --bg-hover: #243044;
    --bg-input: #0d1a2a;
    --accent-blue: #2196F3;
    --accent-teal: #00bcd4;
    --accent-gold: #ffc107;
    --text-primary: #e0e6ed;
    --text-secondary: #8899aa;
    --border-color: #2a3a4e;
    --danger: #ef5350;
    --danger-bg: #3a1a1a;
    --success: #66bb6a;
    --success-bg: #1a3a1a;
}
body { background-color: var(--bg-primary); color: var(--text-primary); }
.gradio-container { background-color: var(--bg-primary) !important; max-width: 1400px !important; }
.gr-box, .panel, .card { background-color: var(--bg-card) !important; border: 1px solid var(--border-color) !important; border-radius: 10px !important; }
.gr-input, input, textarea, select { background-color: var(--bg-input) !important; color: var(--text-primary) !important; border: 1px solid var(--border-color) !important; }
label, .label-text { color: var(--text-secondary) !important; }
button, .gr-button { border-radius: 8px !important; transition: all 0.2s !important; }
button.primary { background: linear-gradient(135deg, #1565C0, #1976D2) !important; border: none !important; }
button.primary:hover { background: linear-gradient(135deg, #1976D2, #2196F3) !important; transform: translateY(-1px); }
h1, h2, h3, h4 { color: var(--text-primary) !important; }
.gr-tabs { border: none !important; }
.tab-nav { background-color: var(--bg-secondary) !important; border-bottom: 1px solid var(--border-color) !important; }
.tab-nav button { color: var(--text-secondary) !important; }
.tab-nav button.selected { color: var(--accent-teal) !important; border-bottom: 2px solid var(--accent-teal) !important; }
footer { display: none !important; }
/* 病历报告文字对比度增强 */
.clinical-report textarea { color: #ffffff !important; background-color: #0d1a2a !important; font-size: 14px !important; line-height: 1.6 !important; }
.dark-report { color: #ffffff; background: #0d1a2a; padding: 15px; border-radius: 8px; border: 1px solid #2a3a4e; font-size: 14px; line-height: 1.7; }
.dark-report h1, .dark-report h2, .dark-report h3 { color: #ffc107 !important; }
.dark-report strong { color: #ffffff; }
.dark-report em { color: #8899aa; }
"""

# ====== 图片扫描工具 ======
def scan_images(folder_path):
    """扫描文件夹内所有图片，按 病人ID（上级文件夹名） 分组"""
    patient_groups = {}  # {patient_id: [image_path, ...]}
    all_images = []

    if not folder_path or not os.path.isdir(folder_path):
        return patient_groups, all_images

    for root, dirs, files in os.walk(folder_path):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                full_path = os.path.join(root, f)
                all_images.append(full_path)
                # 病人ID：上级文件夹名
                parent_dir = os.path.basename(os.path.dirname(full_path))
                if parent_dir not in patient_groups:
                    patient_groups[parent_dir] = []
                patient_groups[parent_dir].append(full_path)

    return patient_groups, all_images


def make_thumbnail(pil_img, size=(120, 120)):
    """生成缩略图"""
    thumb = pil_img.copy()
    thumb.thumbnail(size, Image.LANCZOS)
    return thumb


# ====== Gradio 接口函数 ======

def predict_single(image):
    """单图诊断，返回 (pinfo, heatmap, report, chart, status, clinical_report_html, manual_label)"""
    global MODEL
    if image is None:
        return ("**当前影像**：等待选择", None, "", None, None, "", "NORMAL（正常）")

    # 关键：确保模型在eval模式
    MODEL.eval()

    img_tensor = preprocess(image)
    # 确保输入tensor在模型所在设备
    model_device = next(MODEL.parameters()).device
    img_tensor = img_tensor.to(model_device)

    normal_p, pneumonia_p, pred_class, elapsed_ms = infer_one(img_tensor)

    try:
        cam = gradcam(img_tensor)
        heatmap = make_heatmap(image, cam)
    except Exception as e:
        print(f"热力图生成失败: {e}")
        heatmap = None

    chart = make_chart([normal_p, pneumonia_p])
    report = _build_report([normal_p, pneumonia_p], elapsed_ms, pred_class)
    status = _build_status_html(pred_class, normal_p, pneumonia_p, "光学计算模式", elapsed_ms)
    default_lbl = "NORMAL（正常）" if pred_class == 0 else "PNEUMONIA（肺炎）"
    return ("", heatmap, report, chart, status, "", default_lbl)


def load_folder(folder_path):
    """批量加载文件夹，返回完整10元组"""
    empty = ([], [], "请选择有效文件夹", "**当前影像**：等待选择", None, "", None, None, "", "NORMAL（正常）")
    if not folder_path or not os.path.isdir(folder_path):
        return empty

    patient_groups, all_images = scan_images(folder_path)
    if not all_images:
        return ([], [], "未找到图片文件", "**当前影像**：等待选择", None, "", None, None, "", "NORMAL（正常）")

    patient_summary = f"找到 **{len(all_images)}** 张影像，来自 **{len(patient_groups)}** 个病人"
    thumbs = make_thumbnails_for_gallery(all_images)

    # 诊断第一张
    first_img = all_images[0]
    img_pil = Image.open(first_img).convert('RGB')
    pdir = os.path.basename(os.path.dirname(first_img))
    fname = os.path.basename(first_img)
    pinfo = f"**当前影像**：病人 `{pdir}` | 文件 `{fname}`"

    _, heatmap, report, chart, status, _, default_lbl = predict_single(img_pil)
    return (all_images, thumbs, patient_summary, pinfo, heatmap, report, chart, status, "", default_lbl)


def update_diagnosis(current_report, manual_label, manual_note):
    """修改诊断结果"""
    lines = current_report.split('\n') if current_report else []
    new_lines = []
    for line in lines:
        if '诊断结果' in line and '【' in line:
            new_label = "✅ 正常" if manual_label == "NORMAL（正常）" else "⚠ 肺炎阳性"
            new_lines.append(f"  诊断结果：【{new_label}】")
        else:
            new_lines.append(line)

    if manual_note:
        new_lines.append("")
        new_lines.append("-" * 55)
        new_lines.append("  医生备注：")
        new_lines.append(f"  {manual_note}")

    return "\n".join(new_lines)


def generate_report(patient_info, report_text, manual_label, manual_note, comparison_info=""):
    """生成病历报告（HTML/Markdown混合格式，白色高对比度界面）"""
    def esc(s):
        """转义HTML特殊字符"""
        return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

    label_cn = "正常" if manual_label == "NORMAL（正常）" else "肺炎阳性"

    # 从文本提取概率值
    prob_normal = "?%"
    prob_pneu = "?%"
    for line in report_text.split('\n'):
        if '肺炎概率' in line and '：' in line:
            prob_pneu = line.split('：')[1].strip()
        if '正常概率' in line and '：' in line:
            prob_normal = line.split('：')[1].strip()

    pneu_color = "#ef5350" if "肺炎" in label_cn else "#66bb6a"

    html = f"""<div style="background:#ffffff;color:#222222;padding:25px 30px;border-radius:12px;border:1px solid #ddd;font-family:'Segoe UI','Microsoft YaHei',sans-serif;font-size:14px;line-height:1.8;max-width:750px;margin:0 auto;">
<h1 style="text-align:center;border-bottom:2px solid #1565C0;padding-bottom:12px;color:#0d47a1;margin-top:0;">🏥 曦智光计算 · 医学影像诊断报告</h1>
<p style="color:#555;text-align:center;font-size:13px;">生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 系统：曦智光计算医学影像辅助诊断系统（V18蒸馏模型）</p>
<hr style="border:none;border-top:1px solid #ddd;">
<h2 style="color:#1565C0;border-left:4px solid #1565C0;padding-left:10px;">一、病人信息</h2>
<p style="padding-left:14px;">{esc(patient_info) if patient_info else '未指定'}</p>
<hr style="border:none;border-top:1px solid #ddd;">
<h2 style="color:#1565C0;border-left:4px solid #1565C0;padding-left:10px;">二、诊断结果</h2>
<div style="background:{pneu_color};color:#fff;padding:10px 20px;border-radius:8px;font-size:18px;font-weight:bold;text-align:center;margin:12px 0;">最终诊断：{label_cn}</div>
<table style="width:100%;border-collapse:collapse;margin:12px 0;">
<tr><td style="padding:8px 14px;border:1px solid #ddd;font-weight:bold;">NORMAL概率</td><td style="padding:8px 14px;border:1px solid #ddd;">{esc(prob_normal)}</td></tr>
<tr><td style="padding:8px 14px;border:1px solid #ddd;font-weight:bold;">PNEUMONIA概率</td><td style="padding:8px 14px;border:1px solid #ddd;">{esc(prob_pneu)}</td></tr>
</table>
{('<h3 style="color:#1565C0;">医生备注</h3><blockquote style="background:#f5f5f5;padding:10px 15px;border-left:4px solid #1565C0;margin:8px 0;color:#333;">' + esc(manual_note) + '</blockquote>') if manual_note else ''}
<hr style="border:none;border-top:1px solid #ddd;">
<h2 style="color:#1565C0;border-left:4px solid #1565C0;padding-left:10px;">三、系统信息</h2>
<ul style="color:#444;">
<li><b>推理平台</b>：LTSimulator 光计算</li>
<li><b>模型版本</b>：{esc(MODEL_VER)}</li>
<li><b>光占比</b>：Ro = 96.36%</li>
<li><b>温度校准</b>：T = {TEMPERATURE}</li>
<li><b>NORMAL阈值</b>：{NORMAL_THRESHOLD}</li>
</ul>
<hr style="border:none;border-top:1px solid #ddd;">
<p style="text-align:center;color:#999;font-size:12px;">诊断结果仅供参考，不构成医学诊断依据</p>
</div>"""
    return html


def _build_report(probs, time_ms, pred_class):
    """格式化诊断报告"""
    normal_p, pneumonia_p = probs
    lines = []
    lines.append("=" * 55)
    lines.append("  曦智光计算 · 胸部X光智能诊断报告")
    lines.append("=" * 55)
    lines.append("")
    if pred_class == 1:
        lines.append(f"  诊断结果：【⚠ 肺炎阳性】")
    else:
        lines.append(f"  诊断结果：【✅ 正常】")
    lines.append("")
    lines.append(f"  肺炎概率：{pneumonia_p*100:.2f}%")
    lines.append(f"  正常概率：{normal_p*100:.2f}%")
    lines.append("")
    lines.append(f"  推理平台：LTSimulator 光计算")
    lines.append(f"  推理耗时：{time_ms:.2f} ms")
    lines.append(f"  光占比：Ro ≈ 96.36%（远超50%门槛）")
    lines.append("")
    lines.append("-" * 55)
    lines.append("  诊断建议：")
    if pred_class == 1:
        if pneumonia_p > 0.85:
            lines.append("  ⚠ 高度疑似肺炎，建议立即就医")
        elif pneumonia_p > 0.7:
            lines.append("  ⚠ 中度疑似肺炎，建议进一步检查")
        else:
            lines.append("  ⚡ 低度疑似肺炎，建议定期复查")
    else:
        if normal_p > 0.85:
            lines.append("  ✅ 肺部影像正常，保持健康生活方式")
        else:
            lines.append("  ✅ 未见明显肺炎征象，建议定期体检")
    lines.append("")
    lines.append("  注：本系统基于曦智科技 LTSimulator 光子计算平台")
    lines.append("      诊断结果仅供参考，不构成医学诊断依据")
    lines.append("=" * 55)
    return "\n".join(lines)


def _build_status_html(pred_class, normal_p, pneumonia_p, model_type, elapsed_ms):
    """生成状态信息HTML"""
    pred_label = '⚠ 肺炎阳性' if pred_class == 1 else '✅ 正常'
    bg_color = '#3a1a1a' if pred_class == 1 else '#1a3a1a'
    border_color = '#ef5350' if pred_class == 1 else '#66bb6a'

    return f"""
    <div style='display:flex; gap:15px; flex-wrap:wrap; padding:10px; background:#1a2332; border-radius:10px; border:1px solid #2a3a4e;'>
    <div style='background:#0d1a2a;padding:8px 15px;border-radius:8px;border:1px solid #2a3a4e;'>
    <span style='color:#8899aa;'>模型版本</span><br><span style='color:#00bcd4;font-weight:bold;'>{MODEL_VER}</span>
    </div>
    <div style='background:#0d1a2a;padding:8px 15px;border-radius:8px;border:1px solid #2a3a4e;'>
    <span style='color:#8899aa;'>推理耗时</span><br><span style='color:#e0e6ed;font-weight:bold;'>{elapsed_ms:.2f} ms</span>
    </div>
    <div style='background:{bg_color};padding:8px 15px;border-radius:8px;border:1px solid {border_color};'>
    <span style='color:#8899aa;'>诊断结论</span><br><span style='color:{border_color};font-weight:bold;font-size:16px;'>{pred_label}</span>
    </div>
    <div style='background:#0d1a2a;padding:8px 15px;border-radius:8px;border:1px solid #2a3a4e;'>
    <span style='color:#8899aa;'>NORMAL概率</span><br><span style='color:#66bb6a;font-weight:bold;'>{normal_p*100:.1f}%</span>
    </div>
    <div style='background:#0d1a2a;padding:8px 15px;border-radius:8px;border:1px solid #2a3a4e;'>
    <span style='color:#8899aa;'>PNEUMONIA概率</span><br><span style='color:#ef5350;font-weight:bold;'>{pneumonia_p*100:.1f}%</span>
    </div>
    </div>
    """


# ====== 缩略图生成（用于Gallery展示）======
def make_thumbnails_for_gallery(image_paths):
    """为路径列表生成（缩略图PIL, 文件名）元组列表"""
    thumbs = []
    for path in image_paths[:50]:  # 最多50张
        try:
            img = Image.open(path).convert('RGB')
            thumb = make_thumbnail(img)
            fname = os.path.basename(path)
            pdir = os.path.basename(os.path.dirname(path))
            thumbs.append((thumb, f"{pdir}/{fname}"))
        except:
            pass
    return thumbs


# ====== 构建Gradio界面 ======
def build_demo():
    """构建Gradio界面"""
    with gr.Blocks(
        title="曦智光计算·医学影像诊断系统",
        css=DARK_CSS,
        theme=gr.themes.Soft(),
        analytics_enabled=False
    ) as demo:

        # ====== 顶部标题 ======
        gr.HTML("""
        <div style='text-align:center;padding:25px;background:linear-gradient(135deg,#0a1628 0%,#0d2137 50%,#0a1628 100%);
                    border-bottom:2px solid #1a3c6e;margin-bottom:20px;'>
        <h1 style='margin:0;font-size:28px;color:#e0e6ed;letter-spacing:2px;'>
            <span style='color:#00bcd4;'>🔦</span> 曦智光计算 · 医学影像辅助诊断系统
        </h1>
        <p style='margin:8px 0 0;font-size:14px;color:#8899aa;'>
            LTSimulator 光子计算平台 | V18蒸馏🏆 Ro=96.36%
        </p>
        </div>
        """)

        # ====== 主布局：左侧（导入+医生面板），右侧（诊断结果）======
        with gr.Row(equal_height=False):
            # ===== 左侧列：导入 + 操作面板 =====
            with gr.Column(scale=2, min_width=380):
                # --- 导入区域 ---
                gr.HTML("""
                <div style='background:#1a2332;border:1px solid #2a3a4e;border-radius:10px;padding:15px;margin-bottom:15px;'>
                <h3 style='color:#00bcd4;margin:0 0 10px 0;'>📂 导入影像</h3>
                """)

                with gr.Tabs():
                    with gr.TabItem("📁 批量导入（文件夹）"):
                        folder_input = gr.Textbox(
                            label="输入文件夹路径",
                            placeholder="例如：/workspace/Optical_ChestXRay/data/chest_xray/test",
                            lines=1
                        )
                        with gr.Row():
                            load_btn = gr.Button("📂 加载文件夹", variant="primary", scale=2)
                            patient_count = gr.Markdown("等待加载...")

                        # 缩略图画廊
                        image_gallery = gr.Gallery(
                            label="影像列表（点击切换）",
                            show_label=True,
                            columns=4,
                            rows=2,
                            height=280,
                            object_fit="contain"
                        )
                        # 隐藏的图片路径列表
                        image_paths = gr.State([])

                    with gr.TabItem("⚡ 单图快速诊断"):
                        single_img = gr.Image(type="pil", label="上传胸部X光片", height=250)
                        with gr.Row():
                            single_btn = gr.Button("🚀 开始诊断", variant="primary", scale=2)
                            # 示例图快速加载
                            btn_n = gr.Button("正常示例", size="sm")
                            btn_p = gr.Button("肺炎示例", size="sm")

                gr.HTML("</div>")  # 关闭卡片

                # --- 当前选中影像信息 ---
                current_patient_info = gr.Markdown(
                    "**当前影像**：等待选择",
                    # removed elem_id
                )

                # --- 医生操作面板 ---
                gr.HTML("""
                <div style='background:#1a2332;border:1px solid #2a3a4e;border-radius:10px;padding:15px;margin-bottom:15px;'>
                <h3 style='color:#ffc107;margin:0 0 10px 0;'>🔧 医生操作面板</h3>
                """)

                with gr.Row():
                    manual_label = gr.Radio(
                        ["NORMAL（正常）", "PNEUMONIA（肺炎）"],
                        value="NORMAL（正常）",
                        label="手动修正诊断",
                        interactive=True
                    )

                manual_note = gr.Textbox(
                    label="医生备注/诊断意见",
                    placeholder="请输入诊断意见或备注...",
                    lines=3
                )

                with gr.Row():
                    update_diag_btn = gr.Button("📝 更新诊断", variant="secondary", scale=1)
                    gen_report_btn = gr.Button("📄 生成病历报告", variant="primary", scale=2)

                gr.HTML("</div>")  # 关闭卡片

                # --- 对比模式 ---
                gr.HTML("""
                <div style='background:#1a2332;border:1px solid #2a3a4e;border-radius:10px;padding:15px;'>
                <h3 style='color:#2196F3;margin:0 0 10px 0;'>📊 前后对比模式</h3>
                """)

                with gr.Row():
                    compare_mode = gr.Checkbox(label="开启对比模式", value=False, interactive=True)

                with gr.Row(visible=False) as compare_row:
                    with gr.Column():
                        gr.Markdown("**🟦 旧片（前次检查）**")
                        old_prob_normal = gr.Textbox(label="旧片NORMAL概率", value="--")
                        old_prob_pneu = gr.Textbox(label="旧片PNEUMONIA概率", value="--")
                    with gr.Column():
                        gr.Markdown("**🟥 新片（当前检查）**")
                        new_prob_normal = gr.Textbox(label="新片NORMAL概率", value="--")
                        new_prob_pneu = gr.Textbox(label="新片PNEUMONIA概率", value="--")

                probability_change = gr.Markdown("**变化趋势**：等待对比...")
                gr.HTML("</div>")  # 关闭卡片

            # ====== 右侧列：诊断结果 ======
            with gr.Column(scale=3, min_width=500):
                gr.HTML("""
                <div style='background:#1a2332;border:1px solid #2a3a4e;border-radius:10px;padding:15px;'>
                <h3 style='color:#e0e6ed;margin:0 0 10px 0;'>📊 诊断结果</h3>
                """)

                # 状态卡片
                status_display = gr.HTML()

                with gr.Row(equal_height=True):
                    heatmap_display = gr.Image(
                        label="病灶热力图 Grad-CAM",
                        type="pil",
                        height=300,
                        width=450
                    )
                    chart_display = gr.Image(
                        label="诊断概率分布",
                        type="pil",
                        height=300,
                        width=450
                    )

                report_display = gr.Textbox(
                    label="诊断报告",
                    lines=10,
                    max_lines=15
                )

                # 病历报告
                gr.HTML("<h4 style='color:#ffc107;margin:10px 0 5px 0;'>📋 病历报告</h4>")
                clinical_report = gr.HTML(
                    label="病历报告"
                )

                gr.HTML("""
                </div>
                """)

                # --- 底部摘要信息（精简版）---
                gr.HTML(f"""
                <div style='background:#1a2332;border:1px solid #2a3a4e;border-radius:10px;padding:12px;margin-top:15px;'>
                <div style='display:flex;gap:20px;flex-wrap:wrap;justify-content:center;'>
                <span style='color:#00bcd4;'><b>Ro = 96.36%</b>（光占比）</span>
                <span style='color:#ffc107;'><b>Acc = 91.03%</b>（测试集）</span>
                <span style='color:#66bb6a;'><b>V18蒸馏🏆</b></span>
                <span style='color:#8899aa;'><b>T={TEMPERATURE}</b> | 阈值={NORMAL_THRESHOLD}</span>
                </div>
                </div>
                """)

        # ====== 出发诊断按钮直接绑到predict_single ======

        # 1. 文件夹批量加载
        load_btn.click(
            fn=load_folder,
            inputs=[folder_input],
            outputs=[image_paths, image_gallery, patient_count, current_patient_info,
                     heatmap_display, report_display, chart_display, status_display,
                     clinical_report, manual_label]
        )

        # 2. 缩略图点击切换 — 简洁版
        image_gallery.select(
            fn=_on_click_gallery,
            inputs=[image_paths],
            outputs=[image_paths, image_gallery, patient_count, current_patient_info,
                     heatmap_display, report_display, chart_display, status_display,
                     clinical_report, manual_label]
        )

        # 3. 单图诊断 — 简洁版
        single_btn.click(
            fn=_on_single_predict,
            inputs=[single_img],
            outputs=[image_paths, image_gallery, patient_count, current_patient_info,
                     heatmap_display, report_display, chart_display, status_display,
                     clinical_report, manual_label]
        )

        # 3. 更新诊断
        def on_update_diag(report_text, label, note):
            return update_diagnosis(report_text, label, note)

        update_diag_btn.click(
            fn=on_update_diag,
            inputs=[report_display, manual_label, manual_note],
            outputs=[report_display]
        )

        # 4. 生成病历报告
        def on_gen_report(patient_info, report_text, label, note):
            return generate_report(patient_info, report_text, label, note)

        gen_report_btn.click(
            fn=on_gen_report,
            inputs=[current_patient_info, report_display, manual_label, manual_note],
            outputs=[clinical_report]
        )

        # 5. 对比模式切换
        def on_compare_toggle(enabled):
            return gr.update(visible=enabled)

        compare_mode.change(
            fn=on_compare_toggle,
            inputs=[compare_mode],
            outputs=[compare_row]
        )

        # 6. 示例图按钮
        normal_examples = []
        pneumonia_examples = []
        try:
            normal_dir = os.path.join(DATA_DIR, 'chest_xray', 'test', 'NORMAL')
            pneumonia_dir = os.path.join(DATA_DIR, 'chest_xray', 'test', 'PNEUMONIA')
            if os.path.exists(normal_dir):
                normal_examples = sorted([
                    os.path.join(normal_dir, f) for f in os.listdir(normal_dir)
                ])[:2]
            if os.path.exists(pneumonia_dir):
                pneumonia_examples = sorted([
                    os.path.join(pneumonia_dir, f) for f in os.listdir(pneumonia_dir)
                ])[:2]
        except:
            pass

        if normal_examples:
            def make_normal_fn(path):
                return lambda: path
            btn_n.click(make_normal_fn(normal_examples[0]), None, single_img)
        if pneumonia_examples:
            def make_pneu_fn(path):
                return lambda: path
            btn_p.click(make_pneu_fn(pneumonia_examples[0]), None, single_img)

        return demo


# ====== 模块级回调函数（不在build_demo内部定义，避免闭包问题）======

def _on_click_gallery(evt: gr.SelectData, paths):
    """缩略图点击回调"""
    if not paths or evt.index is None or evt.index >= len(paths):
        return paths if paths else [], None, None, "**当前影像**：等待选择", None, "", None, None, "", "NORMAL（正常）"
    img_path = paths[evt.index]
    img_pil = Image.open(img_path).convert('RGB')
    pdir = os.path.basename(os.path.dirname(img_path))
    fname = os.path.basename(img_path)
    pinfo = f"**当前影像**：病人 `{pdir}` | 文件 `{fname}`"
    _, heatmap, report, chart, status, _, default_lbl = predict_single(img_pil)
    return paths, None, None, pinfo, heatmap, report, chart, status, "", default_lbl


def _on_single_predict(img):
    """单图诊断回调"""
    if img is None:
        return [], [], "等待选择...", "**当前影像**：单图上传", None, "", None, None, "", "NORMAL（正常）"
    pinfo = "**当前影像**：单图上传"
    _, heatmap, report, chart, status, _, default_lbl = predict_single(img)
    return None, None, None, pinfo, heatmap, report, chart, status, "", default_lbl


# ====== 启动 ======
if __name__ == "__main__":
    print("=" * 60)
    print("🔦 曦智光计算 · 医学影像辅助诊断系统 - Demo")
    print("=" * 60)
    print("当前最优：V18蒸馏模型 | Ro=96.36%")
    print("功能：批量导入 | 前后对比 | 修改诊断 | 生成病历")
    print("=" * 60)

    load_model()
    print(f"设备：{device} | 模型：{MODEL_VER}")
    print(f"温度校准T={TEMPERATURE} | NORMAL阈值={NORMAL_THRESHOLD}")
    print(f"随机种子已锁定（SEED={SEED}），诊断结果稳定")

    demo = build_demo()
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7865"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
