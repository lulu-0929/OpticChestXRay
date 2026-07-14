# 🔦 曦智科技 · 光计算加速医学影像诊断系统

> **集创赛 · 分赛区决赛作品** | 曦智科技企业命题 | 医疗健康赛道
> **最终模型**：OpticalChestXRayV8 | **测试集 Acc：87.02% ✅ | 光占比 Ro：94.72% ✅ | G评分：通过（17.98/30）**

---

## 📋 项目简介

基于 **LTSimulator v1.4.6 光子计算模拟器平台**，设计一套胸部X光肺炎诊断系统。核心创新为用 `nn.Linear` 模拟光计算卷积/池化全流程，实现 **光占比 Ro > 94%** 的光子计算诊断方案。

- **任务**：胸部X光肺炎二分类（正常 NORMAL / 肺炎 PNEUMONIA）
- **输入**：128×128 灰度胸部X光片
- **数据集**：[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **数据分布**：NORMAL=1341 | PNEUMONIA=3875 | 测试集=624

---

## 🏗️ 完整项目结构

```
OpticChestXRay/
├── src/                              # 源代码目录
│   ├── model.py                      # V1-V5 基准模型定义（4层光卷积+2层FC）
│   ├── train.py                      # V1-V4 基础训练脚本（CrossEntropyLoss + Adam）
│   ├── train_v5.py                   # V5 优化训练（WeightedRandomSampler + 加权损失）
│   ├── train_v6.py                   # V6 重构训练（FC分解8192→512→128→2 + 标签平滑 + TTA）
│   ├── train_v7.py                   # V7 针对性优化（FocalLoss + FC精简 + 分层采样）
│   ├── train_v8.py                   # 🏆 V8 最终版（FocalLossγ=3.0 + 过采样5x + TTA15 + 温度校准）
│   ├── config.py                     # 配置文件（数据集路径、超参数等）
│   ├── utils.py                      # 工具函数（评估、可视化、报告生成）
│   ├── infer.py                      # 推理评估脚本
│   ├── app_demo.py                   # Gradio 交互式Demo（V8架构）
│   ├── app_streamlit.py              # Streamlit 交互式Demo
│   ├── patch_v8.py                   # V8补丁脚本
│   ├── write_v8.py                   # V8写入测试
│   │
│   └── optimized/                    # V8之后迭代探索版本目录
│       ├── train_v9.py               # ❌ V9 - MultiClassAsymmetricLoss梯度消失（82.05%）
│       ├── train_v9_bak.py           # V9备份版本
│       ├── train_v10.py              # ❌ V10 - 大FC架构未提升（86.86%，略低于V8）
│       ├── train_v10_fix.py          # V10修复版（CrossEntropy + 阈值校准 + Mixup）
│       ├── train_v11.py              # ⏳ V11 - 迁移学习ResNet18 + CutMix（训练中）
│       └── debug_train.py            # 调试训练脚本
│
├── output/                           # 训练输出和报告
│   ├── best_optical_v8.pth           # 🏆 V8 最佳模型权重（13MB，但git LFS存储）
│   ├── optical_v5_report.txt         # V5 训练报告
│   ├── optical_v6_report.txt         # V6 训练报告（83.33%）
│   ├── optical_v7_report.txt         # V7 训练报告（84.78%）
│   ├── optical_v8_report.txt         # V8 最终训练报告（87.02% ✅）
│   ├── optical_v9_report.txt         # V9 训练报告（82.05% ❌）
│   ├── optical_v10_report.txt        # V10 训练报告（86.86%）
│   ├── optical_v11_report.txt        # V11 训练报告（开始阶段）
│   └── optical_chestxray_report.txt  # 项目汇总报告
│
├── doc/                              # 文档目录
│   ├── 问题记录.md                    # 开发踩坑记录（完整问题分析）
│   ├── 模型演进纪实.md                 # V1-V11详细演进过程（架构图+失败分析）
│   └── 5代模型性能分析报告.docx         # V1-V5详细分析报告
│
├── docker/                           # Docker 容器配置
│   └── README.md                     # 容器使用说明
│
├── .gitignore                        # Git忽略规则
└── README.md                         # 本文件
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

- 测试集：**79.33%**（NORMAL 仅 48.3%）

### V6（架构重构+强正则化）— Ro最高版

**核心改进**：FC 层分解 8192→**512→128**→2，Dropout=0.5，标签平滑 LabelSmoothing(0.1)，ReduceLROnPlateau，TTA 5次投票

- Ro=**97.23%**（所有版本中最高光占比）
- 测试集：**83.33%**（NORMAL 58.1%，PNEUMONIA 98.5%）

### V7（针对性优化）

**核心改进**：FC 精简 8192→**256→64**→2，**Focal Loss (γ=2.0)**，验证集分层采样(NORMAL≥35%)，NORMAL 过采样 3x

- 测试集：**84.78%**（NORMAL 60.3%）

### V8 🏆（最终版）— G评分通过 ✅

**核心改进**：
1. FC层 8192→**384→64**→2，参数量 3.27M
2. **Focal Loss (γ=3.0)** + 标签平滑 0.05
3. **NORMAL 过采样 5x**（NORMAL=5030, PNEUMONIA=3624）
4. **温度校准 T=1.5** + **TTA 15次投票**
5. **加权模型选择**（0.7×Acc + 0.3×NORMAL_Recall）

**最终结果**：
- **测试集 Acc：87.02% ✅（超85%门槛）**
- NORMAL 召回率：**68.8%** / PNEUMONIA 召回率：**97.9%**
- **G 评分：通过（总分17.98/30）** ✅
- 训练耗时：13,725s（~3.8h）

### V9-V11（V8之后迭代探索）

| 版本 | 尝试方向 | 结果 | 失败原因 |
|------|---------|------|---------|
| V9 | MultiClassAsymmetricLoss + 大FC | 82.05% ❌ | 梯度消失，全判PNEUMONIA |
| V10 | V6大FC + V8全部优化 | 86.86% | 大FC未带来提升，数据量不足 |
| V11 | 迁移学习ResNet18 + CutMix | ⏳ 训练中 | 冻结前2层导致NORMAL全错 |

> 详细演进过程见 [doc/模型演进纪实.md](doc/模型演进纪实.md)

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

---

## 光计算层清单（V8）

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

## 🚀 快速开始

### 环境要求

- Docker 容器（推荐使用已配置好的 `reverent_cannon`）
- Python 3.9 + PyTorch 2.1 + torchvision 0.16
- GPU（可选，生产环境建议使用 CUDA 加速）

### Docker 容器配置

```bash
# 创建容器（基于 LTSimulator v1.4.6 镜像）
docker run -d --gpus all --name reverent_cannon \
  lightelligence.docker:lt-simulator_v1.4.6-final sleep infinity

# 激活 moca_llm 环境
docker exec reverent_cannon bash -c "source /local/miniconda/bin/activate moca_llm"
```

### 训练

```bash
# 直接执行（单行命令）
docker exec reverent_cannon bash -c "cd /workspace/Optical_ChestXRay && /local/miniconda/envs/moca_llm/bin/python src/train_v8.py"
```

### 启动 Demo

```bash
docker exec reverent_cannon bash -c "cd /workspace/Optical_ChestXRay && /local/miniconda/envs/moca_llm/bin/python src/app_demo.py"
```

Demo 使用 Gradio 框架，启动后会自动生成公网可访问链接。

### 数据准备

1. 从 [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 下载数据集
2. 解压到 `data/chest_xray/` 目录（训练集/验证集/测试集结构保持不变）
3. 训练脚本会自动检测数据目录

> **注意**：数据集较大（~1.5GB），最终 commit 中仅保存数据加载代码，不包含原始图片。

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

## 🐳 容器技术栈

| 项目 | 值 |
|------|-----|
| 容器名 | `reverent_cannon` |
| 镜像 | `lightelligence.docker:lt-simulator_v1.4.6-final` |
| Python 版本 | 3.9.5（moca_llm 环境） |
| PyTorch | 2.1.0+cu121（CUDA 12.1）|
| torchvision | 0.16.0+cu121 |
| GPU | RTX 4070 Laptop GPU（8GB VRAM） |
| 容器工作目录 | `/workspace/Optical_ChestXRay/` |

> 详细配置说明见 [docker/README.md](docker/README.md)

---

## ⚠️ 常见问题

### Q: Demo 启动报 `localhost not accessible`
**解决**：Demo 已设置 `share=True`，启动后会自动生成 Gradio 公网链接。

### Q: 容器内无法访问外网
**解决**：检查 Docker 网络配置，容器需要访问 huggingface 下载 Gradio 资源。

### Q: 模型推理慢
**解决**：当前在 CPU 上运行（约 3-4ms/张）。如果在支持 CUDA 的环境下运行，可启用 GPU 加速。

### Q: Windows 路径问题
**解决**：Git Bash 中容器内路径加双斜杠 `//` 避免被 Git Bash 转换为 Windows 路径。

---

## 📝 待优化方向

1. **增加卷积层光计算占比**，减少 FC 层过度依赖
2. **探索 LTSimulator 真实光学组件调用**（非 Linear 模拟）
3. **拓展多种医学影像支持**（CT、MRI）
4. **模型量化 + ONNX 部署**，提升推理速度
5. **迁移学习微调策略优化**：解冻更多层+更精细的学习率调度
6. **数据增强对比实验**：CutMix/Mixup 在光计算模型下的最优配置

---

## 📄 许可证

本项目为集创赛 · 曦智科技企业命题参赛作品。
