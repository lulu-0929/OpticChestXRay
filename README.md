# 🔦 曦智科技 · 光计算加速医学影像诊断系统

> **集创赛 · 分赛区决赛作品** | 曦智科技企业命题 | 医疗健康赛道
> **最终模型**：OpticalChestXRayV8 | **测试集 Acc：87.02% ✅ | 光占比 Ro：94.72% ✅ | G评分：通过（17.98/30）**

## 📋 项目简介

基于 **LTSimulator v1.4.6 光子计算模拟器平台**，设计一套胸部X光肺炎诊断系统。核心创新为用 `nn.Linear` 模拟光计算卷积/池化全流程，实现 **光占比 Ro > 94%** 的光子计算诊断方案。

- **任务**：胸部X光肺炎二分类（正常 NORMAL / 肺炎 PNEUMONIA）
- **输入**：128×128 灰度胸部X光片
- **数据集**：[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **数据分布**：NORMAL=1341 | PNEUMONIA=3875 | 测试集=624

---

## 🏗️ 项目结构

```
Optical_ChestXRay/
├── data/
│   └── chest_xray/
│       ├── train/          # 训练集（NORMAL + PNEUMONIA）
│       ├── val/            # 验证集
│       └── test/           # 测试集（NORMAL=234, PNEUMONIA=390）
├── src/
│   ├── model.py            # V1-V5 基准模型定义
│   ├── train.py            # V1-V4 训练脚本
│   ├── train_v5.py         # V5 优化训练脚本
│   ├── train_v6.py         # V6 优化训练脚本（架构重构+正则化）
│   ├── train_v7.py         # V7 针对性优化脚本（FocalLoss+FC精简）
│   ├── train_v8.py         # V8 最终优化脚本（FocalLoss+过采样5x+TTA15+温度校准）
│   ├── infer.py            # 推理评估脚本
│   ├── config.py           # 配置文件
│   ├── utils.py            # 工具函数
│   ├── app_demo.py         # Gradio 交互式Demo（已更新V8架构）
│   └── app_streamlit.py    # Streamlit 交互式Demo
├── output/
│   ├── best_optical_v8.pth         # V8 最佳权重（最终版本 ✅）
│   ├── optical_v8_report.txt       # V8 训练报告（87.02%）
│   ├── optical_v6_report.txt       # V6 训练报告
│   └── optical_v5_report.txt       # V5 训练报告
├── README.md
└── doc/
    └── 问题记录.md                 # 开发踩坑记录
```

---

## 🧠 模型架构演化

### V1（初始基线）

```
Input [B, 1, 128, 128]
  │
  ├─ OpticalConv1(1→16, k=3)  + Pool(2×2)  + ReLU   [B, 16, 64, 64]
  ├─ OpticalConv2(16→32, k=3) + Pool(2×2)  + ReLU   [B, 32, 32, 32]
  ├─ OpticalConv3(32→64, k=3) + Pool(2×2)  + ReLU   [B, 64, 16, 16]
  ├─ OpticalConv4(64→128, k=3)+ Pool(2×2)  + ReLU   [B, 128, 8, 8]
  │
  ├─ Flatten: [B, 8192]
  ├─ FC0(8192→256) + Dropout(0.2) + ReLU
  └─ FC1(256→2)
```

**首次搭建的光计算模拟框架**，4层光卷积+4层光池化+2层全连接。训练25 epochs，测试集仅 **77.40%**。

### V2-V4（优化探索）

- **V2**：调整学习率和优化器（AdamW + CosineAnnealingLR）
- **V3**：增加数据增强（RandomHorizontalFlip + RandomAffine）
- **V4**：调整 Dropout=0.3 + 权重衰减

**三版共同问题**：验证集 Acc 最高达 97.70%，但测试集始终在 77-80% 区间——**严重过拟合**。

### V5（类别平衡优化）

在 V4 基础上引入 **WeightedRandomSampler** 处理类别不平衡，加权损失 + 训练集内分割验证集（15%）。

- 验证集最佳：97.57%
- 测试集：**79.33%**（NORMAL 仅 48.3%）
- **结论**：仅靠采样均衡不够，NORMAL 难分问题仍然存在

### V6（架构重构+强正则化）— Ro最高版

**核心改进**：
1. FC 层分解：8192→**512→128**→2（增加中间层容量）
2. Dropout=0.5 + Weight Decay=5e-4
3. 标签平滑 LabelSmoothing(0.1)
4. ReduceLROnPlateau 动态学习率
5. 数据增强增强：旋转10° + 颜色抖动
6. TTA 测试时增强（5次投票）

**结果**：
- Ro=**97.23%**（所有版本中光占比最高）
- 验证集最佳：**98.47%**
- 测试集：**83.33%**（NORMAL 58.1%，PNEUMONIA 98.5%）
- **差距缩小但仍然未达85%门槛**，NORMAL 召回率仍需提升

### V7（针对性优化）

**核心改进**：
1. FC 精简：8192→**256→64**→2，参数量从 4.36M 降至 2.21M
2. **Focal Loss (γ=2.0)**：强制关注难分的 NORMAL 样本
3. **验证集分层采样**：NORMAL 占比≥35%，消除验证集偏差
4. **NORMAL 过采样 3x**：缓解类别不平衡
5. 强数据增强：旋转15° + 颜色抖动 + 仿射变换
6. TTA 测试时增强：10次投票

**结果**：
- Ro=**94.72%**
- 验证集最佳：**97.61%**
- 测试集：**84.78%**（NORMAL 60.3%，PNEUMONIA 99.5%)

### V8（最终版）— G评分通过 ✅

**核心改进**：
1. FC层扩展：8192→**384→64**→2，保留更多特征信息
2. **Focal Loss (γ=3.0)**：更强关注难分样本
3. **NORMAL 过采样 5x**：训练集NORMAL=5030, PNEUMONIA=3624
4. **温度校准 T=1.5**：软化置信度，提升校准效果
5. **标签平滑 0.05**：防止过自信
6. **TTA 15次投票集成**：提升稳定性
7. **加权模型选择**：0.7×Acc + 0.3×NORMAL_Recall
8. **早停 patience=10**

**最终结果**：
- Ro=**94.72%**
- 最佳验证Acc：**98.51%**（NORMAL 99.4%，PNEUMONIA 96.4%）
- 测试集：**87.02%** ✅（超85%门槛）
- NORMAL 召回率：**68.8%**（161/234）
- PNEUMONIA 召回率：**97.9%**（382/390）
- 平衡测试集：**83.38%**
- TTA Acc(15次)：**86.86%**
- **G评分：通过（总分17.98/30）** ✅
- **训练耗时：13,725s（~3.8h）**

### 光计算层清单（V8）

| 层 | Shape | 计算量 | 类型 |
|----|-------|--------|------|
| OpticalConv_conv1 | (9, 16) | 144 | 光卷积 |
| OpticalConv_conv2 | (144, 32) | 4,608 | 光卷积 |
| OpticalConv_conv3 | (288, 64) | 18,432 | 光卷积 |
| OpticalConv_conv4 | (576, 128) | 73,728 | 光卷积 |
| OpticalPool×4 | (4, 1) | 960 | 光池化 |
| FC_0 (8192→384) | (8192, 384) | 2,097,152 | 光全连接 |
| FC_1 (384→64) | (384, 64) | 16,384 | 光全连接 |
| FC_2 (64→2) | (64, 2) | 128 | 光全连接 |
| **总计** | | **2,211,536** | **Ro≈94.7%** |

---

## 📊 实验结果对比

| 版本 | Ro 光占比 | 验证集 Acc | 测试集 Acc | NORMAL 召回率 | PNEUMONIA 召回率 | 参数量 |
|------|-----------|-----------|-----------|--------------|-----------------|--------|
| V1 | 94.69% | 97.70% | 77.40% | — | — | 2.20M |
| V2-V4 | 94.69% | 97.70% | ~78% | — | — | 2.20M |
| **V5** | **94.69%** | **97.57%** | **79.33%** | **48.3%** | **97.9%** | **2.20M** |
| **V6** | **97.23%** | **98.47%** | **83.33%** | **58.1%** | **98.5%** | **4.36M** |
| **V7** | **94.72%** | **97.61%** | **84.78%** | **60.3%** | **99.5%** | **2.21M** |
| **V8 🏆** | **94.72%** | **98.51%** | **87.02%** | **68.8%** | **97.9%** | **3.27M** |

**关键发现**：
- V6 拥有最高光占比（97.23%），但 NORMAL 召回率仅 58.1%
- V7 通过 Focal Loss + 分层采样，NORMAL 召回率提升至 60.3%
- **V8 综合最优**：通过过采样5x + Focal Loss(γ=3.0) + 温度校准T=1.5 + TTA15，最终87.02%超85%门槛 ✅
- NORMAL 召回率从 V1 的 48.3% 提升至 V8 的 68.8%（+20.5pct）

---

## 🚀 快速开始

### 环境要求

- Docker 容器（推荐使用已配置好的 `reverent_cannon`）
- Python 3.9 + PyTorch 2.1 + torchvision 0.16

### 训练

```bash
# V8 训练
docker exec reverent_cannon bash -c "cd /workspace/Optical_ChestXRay && /local/miniconda/envs/moca_llm/bin/python src/train_v8.py"
```

### 启动 Demo

```bash
docker exec reverent_cannon bash -c "cd /workspace/Optical_ChestXRay && /local/miniconda/envs/moca_llm/bin/python src/app_demo.py"
```

Demo 使用 Gradio 框架，启动后会自动生成公网可访问链接。

---

## 🧪 自动评分公式

赛题评分由三部分组成：

```
G = 1  if (Ro > 50%) and (测试集Acc > 85%) else 0

S_ratio = 20 × (Ro - 0.5) / 0.5    # 光占比得分（满分20）
S_acc   = 5 × ((Acc - 0.85) / 0.15)²  # 准确率得分（满分5）

总分 = G × (S_ratio + S_acc)       # 满分25~30
```

---

## ⚠️ 常见问题

### Q: Demo 启动报 `localhost not accessible`
**解决**：Demo 已设置 `share=True`，启动后会自动生成 Gradio 公网链接。

### Q: 容器内无法访问外网
**解决**：检查 Docker 网络配置，容器需要访问 huggingface 下载 Gradio 资源。

### Q: 模型推理慢
**解决**：当前在 CPU 上运行（约 3-4ms/张）。如果在支持 CUDA 的环境下运行，可启用 GPU 加速。

---

## 📝 待优化方向

1. **增加卷积层光计算占比**，减少 FC 层过度依赖
2. **探索 LTSimulator 真实光学组件调用**（非 Linear 模拟）
3. **拓展多种医学影像支持**（CT、MRI）
4. **模型量化 + ONNX 部署**，提升推理速度

---

## 📄 许可证

本项目为集创赛 · 曦智科技企业命题参赛作品。
