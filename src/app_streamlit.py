"""
⚠️ 此文件已弃用 — 旧版 Streamlit Demo（V6架构）
================================================
请使用 app_demo_v2.py（Gradio + V18蒸馏模型，功能更全）
基于 LTSimulator 光子计算平台，展示诊断功能
"""
import os, sys, time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# ====== 路径 ======
BASE_DIR = '/workspace/Optical_ChestXRay'
MODEL_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128

# ====== 光计算模型（与V6一致）=======
class OpticalConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super().__init__()
        self.in_c, self.out_c, self.k, self.s, self.p = in_c, out_c, k, s, p
        kh = k if isinstance(k, tuple) else (k, k)
        self.opt = nn.Linear(in_c * kh[0] * kh[1], out_c, bias=True)
        self.unfold = nn.Unfold(kernel_size=kh, stride=s, padding=p)
    def forward(self, x):
        b, c, h, w = x.shape
        kh = self.k if isinstance(self.k, tuple) else (self.k, self.k)
        p = self.unfold(x).transpose(1,2).contiguous()
        out = self.opt(p)
        oh = (h+2*self.p-kh[0])//self.s+1
        ow = (w+2*self.p-kh[1])//self.s+1
        return out.transpose(1,2).view(b,self.out_c,oh,ow)

class OpticalPool2d(nn.Module):
    def __init__(self, c, k=2):
        super().__init__()
        self.c, self.k = c, k
        self.opt = nn.Linear(k*k, 1, bias=False)
    def forward(self, x):
        b,c,h,w = x.shape
        u = x.unfold(2,self.k,self.k).unfold(3,self.k,self.k)
        u = u.contiguous().view(b,c,-1,self.k*self.k)
        p = self.opt(u).squeeze(-1)
        return p.view(b,c,h//self.k,w//self.k)

class Model(nn.Module):
    def __init__(self, in_c=1, n_class=2, drop=0.5):
        super().__init__()
        self.c1=OpticalConv2d(in_c,16,3,p=1); self.p1=OpticalPool2d(16)
        self.c2=OpticalConv2d(16,32,3,p=1); self.p2=OpticalPool2d(32)
        self.c3=OpticalConv2d(32,64,3,p=1); self.p3=OpticalPool2d(64)
        self.c4=OpticalConv2d(64,128,3,p=1); self.p4=OpticalPool2d(128)
        self.fc0=nn.Linear(128*8*8,512); self.bn0=nn.BatchNorm1d(512); self.d0=nn.Dropout(drop)
        self.fc1=nn.Linear(512,128); self.bn1=nn.BatchNorm1d(128); self.d1=nn.Dropout(drop)
        self.fc2=nn.Linear(128,n_class)
    def forward(self, x):
        x=self.p1(self.c1(x)); x=F.relu(x)
        x=self.p2(self.c2(x)); x=F.relu(x)
        x=self.p3(self.c3(x)); x=F.relu(x)
        x=self.p4(self.c4(x)); x=F.relu(x)
        x=x.reshape(x.size(0),-1)
        x=self.fc0(x); x=self.bn0(x); x=self.d0(x); x=F.relu(x)
        x=self.fc1(x); x=self.bn1(x); x=self.d1(x); x=F.relu(x)
        return self.fc2(x)

def gradcam(model, img_t):
    """Grad-CAM热力图"""
    img_t = img_t.to(device); img_t.requires_grad_()
    x = model.c1(img_t); x = model.p1(x); x = F.relu(x)
    x = model.c2(x); x = model.p2(x); x = F.relu(x)
    x = model.c3(x); x = model.p3(x); x = F.relu(x)
    x = model.c4(x)
    feat = x
    x = model.p4(x); x = F.relu(x)
    x = x.reshape(x.size(0),-1)
    x = model.fc0(x); x = model.bn0(x); x = F.relu(x)
    x = model.fc1(x); x = model.bn1(x); x = F.relu(x)
    x = model.fc2(x)
    target = x.argmax(1).item()
    model.zero_grad(); x[0,target].backward()
    w = feat.grad.mean(dim=(2,3), keepdim=True)
    cam = F.relu((w*feat).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE,IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    return (cam-cam.min())/max(cam.max()-cam.min(),1e-8)


# ====== 加载模型 ======
@st.cache_resource
def load_model():
    model = Model().to(device); model.eval()
    candidates = [
        ('V6', os.path.join(MODEL_DIR, 'best_optical_v6.pth')),
        ('V5', os.path.join(MODEL_DIR, 'best_optical_v5.pth')),
        ('V4', os.path.join(MODEL_DIR, 'best_optical_chestxray_v4.pth')),
    ]
    loaded_ver = '未训练'
    for ver, path in candidates:
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location=device), strict=False)
                loaded_ver = ver; break
            except: pass
    return model, loaded_ver


# ====== Streamlit 界面 ======
st.set_page_config(
    page_title="曦智光计算·医学影像诊断",
    page_icon="🔦",
    layout="wide"
)

# 标题
st.markdown("""
<div style='text-align:center;padding:20px;background:linear-gradient(135deg,#1a3c6e,#2d5aa0);color:white;border-radius:15px;margin-bottom:20px;'>
<h1 style='margin:0;font-size:30px;'>🔦 曦智科技 · 光计算加速医学影像诊断系统</h1>
<p style='margin:8px 0 0;font-size:16px;'>LTSimulator 光子计算平台 | 光占比 Ro≈97% | 赛题：医疗健康</p>
</div>
""", unsafe_allow_html=True)

model, model_ver = load_model()

# 侧边栏
st.sidebar.title("📌 系统信息")
st.sidebar.info(f"""
- **模型版本**: {model_ver}
- **设备**: {device}
- **光占比**: Ro≈97%
- **输入尺寸**: 128×128 灰度
- **任务**: 胸部X光肺炎诊断
""")

st.sidebar.title("📂 示例图片")
normal_dir = os.path.join(DATA_DIR, 'chest_xray', 'test', 'NORMAL')
pneu_dir = os.path.join(DATA_DIR, 'chest_xray', 'test', 'PNEUMONIA')

example_files = []
if os.path.exists(normal_dir):
    for f in sorted(os.listdir(normal_dir))[:3]:
        example_files.append(('正常', os.path.join(normal_dir, f)))
if os.path.exists(pneu_dir):
    for f in sorted(os.listdir(pneu_dir))[:3]:
        example_files.append(('肺炎', os.path.join(pneu_dir, f)))

selected_example = st.sidebar.selectbox(
    "选择示例图片",
    [f"{label}: {os.path.basename(p)}" for label, p in example_files]
)

# 主界面
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📤 上传 X 光片")
    uploaded = st.file_uploader("选择胸部X光片 (JPEG/PNG)", type=['jpg', 'jpeg', 'png'])

    # 加载示例或上传
    if not uploaded and selected_example:
        idx = [f"{label}: {os.path.basename(p)}" for label, p in example_files].index(selected_example)
        pil_img = Image.open(example_files[idx][1]).convert('L')
        st.caption(f"示例：{example_files[idx][0]} - {os.path.basename(example_files[idx][1])}")
    elif uploaded:
        pil_img = Image.open(uploaded).convert('L')
    else:
        st.info("请上传图片或选择示例")
        pil_img = None

    if pil_img:
        st.image(pil_img, caption="原始 X 光片", width=300)

        if st.button("🚀 开始诊断", type="primary", use_container_width=True):
            with st.spinner("正在进行光计算推理..."):
                from torchvision import transforms
                t = transforms.Compose([
                    transforms.Resize((IMG_SIZE,IMG_SIZE)),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5],[0.5])
                ])
                img_t = t(pil_img).unsqueeze(0)

                start = time.time()
                with torch.no_grad():
                    out = model(img_t)
                    probs = F.softmax(out, dim=1)
                    p_pneu = probs[0,1].item()
                    p_norm = probs[0,0].item()
                elapsed_ms = (time.time()-start)*1000

                pred = 1 if p_pneu > p_norm else 0

                # 存session_state
                st.session_state['probs'] = [p_norm, p_pneu]
                st.session_state['pred'] = pred
                st.session_state['time'] = elapsed_ms
                st.session_state['pil'] = pil_img
                st.session_state['img_t'] = img_t
                st.session_state['done'] = True

with col2:
    st.subheader("📊 诊断结果")

    if st.session_state.get('done'):
        probs = st.session_state['probs']
        pred = st.session_state['pred']
        t_ms = st.session_state['time']
        pil_img = st.session_state['pil']

        # 状态卡片
        cols = st.columns(4)
        cols[0].metric("模型版本", model_ver)
        cols[1].metric("推理耗时", f"{t_ms:.1f}ms")
        cols[2].metric("光占比", "≈97%")
        cols[3].metric("诊断结论", "⚠ 肺炎" if pred else "✅ 正常",
                       delta_color="off")

        # 概率条
        st.progress(probs[1], text=f"肺炎概率: {probs[1]*100:.1f}%")
        st.progress(probs[0], text=f"正常概率: {probs[0]*100:.1f}%")

        # 热力图
        img_t = st.session_state['img_t']
        cam = gradcam(model, img_t)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        img_rgb = pil_img.resize((IMG_SIZE,IMG_SIZE)).convert('RGB')
        axes[0].imshow(np.array(img_rgb), cmap='gray')
        axes[0].set_title('原始 X 光片'); axes[0].axis('off')
        axes[1].imshow(cam, cmap='jet', alpha=0.7)
        axes[1].set_title('Grad-CAM 病灶热力图', color='red', fontweight='bold'); axes[1].axis('off')
        axes[2].imshow(np.array(img_rgb), cmap='gray', alpha=0.6)
        axes[2].imshow(cam, cmap='jet', alpha=0.5)
        axes[2].set_title('叠加效果'); axes[2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # 诊断报告
        st.markdown("### 📋 诊断报告")
        report = f"""
        <div style='background:#f0f2f6;padding:15px;border-radius:10px;font-family:monospace;'>
        <pre>
═══════════════════════════════════════════
  曦智光计算 · 胸部X光智能诊断报告
═══════════════════════════════════════════

  诊断结果：【{'⚠ 肺炎阳性' if pred else '✅ 正常'}】

  肺炎概率：{probs[1]*100:.2f}%
  正常概率：{probs[0]*100:.2f}%

  推理平台：LTSimulator 光子计算
  推理耗时：{t_ms:.2f} ms
  光占比：Ro≈97%（远超50%门槛）

───────────────────────────────────────
  诊断建议：
  {'⚠ 建议立即就医' if pred and probs[1]>0.85 else '⚠ 建议进一步检查' if pred else '✅ 肺部影像正常'}

  注：本系统基于曦智科技 LTSimulator
      仅供参考，不构成医学诊断依据
═══════════════════════════════════════════
        </pre>
        </div>
        """
        st.markdown(report, unsafe_allow_html=True)
    else:
        st.info("👈 左侧上传 X 光片后点击「开始诊断」")

# 底部：光计算层信息
st.markdown("---")
with st.expander("🔦 光计算层详细清单（光占比 Ro≈97%）"):
    cols = st.columns([1,2,1,1,1])
    cols[0].markdown("**序号**")
    cols[1].markdown("**层名称**")
    cols[2].markdown("**Shape**")
    cols[3].markdown("**计算量**")
    cols[4].markdown("**类型**")
    st.markdown("---")
    layers = [
        ('1','OpticalConv_conv1','(9,16)','144','光卷积'),
        ('2','OpticalConv_conv2','(144,32)','4,608','光卷积'),
        ('3','OpticalConv_conv3','(288,64)','18,432','光卷积'),
        ('4','OpticalConv_conv4','(576,128)','73,728','光卷积'),
        ('5-8','OpticalPool×4','(4,1)','960','光池化'),
        ('9','FC_0(8192→512)','(8192,512)','4,194,304','光全连接'),
        ('10','FC_1(512→128)','(512,128)','65,536','光全连接'),
        ('11','FC_2(128→2)','(128,2)','256','光全连接'),
    ]
    for idx, name, shape, ops, ltype in layers:
        c = st.columns([1,2,1,1,1])
        c[0].write(idx); c[1].write(name); c[2].write(shape); c[3].write(ops); c[4].write(ltype)
    st.success("光计算总量：4,357,712 | 光占比 Ro≈97%")
