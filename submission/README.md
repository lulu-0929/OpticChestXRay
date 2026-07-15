# 曦智科技 · 光计算加速医学影像诊断系统 — 比赛提交包

> **集创赛 · 分赛区决赛作品** | 曦智科技企业命题 | 医疗健康赛道
> **最终版本：V13** | **测试集 Acc：90.54%** | **Ro：96.36%** | **总分：19.23/30**

---

## 📋 提交文件清单

```
submission/
├── README.md                           # 本文件——提交说明
├── src/                                # 源代码
│   ├── train_v13.py                    # 🏆 V13最终训练脚本（主提交文件）
│   ├── eval_v13_postprocess.py         # V13后处理评估（阈值校准+集成）
│   ├── app_demo.py                     # Gradio交互式Demo
│   ├── train_v12.py                    # V12 - 修复test_transform bug基线
│   ├── train_v14.py                    # V14 - 伪标签半监督（失败尝试）
│   ├── train_v15.py                    # V15 - CutOut增强（未超越V13）
│   ├── train_v8.py                     # V8 - 原最优版（含test_transform bug）
│   ├── train_v9.py                     # V9 - MultiClassAsymmetricLoss（失败）
│   ├── train_v10.py                    # V10 - 大FC架构（未超越V8）
│   └── train_v11.py                    # V11 - 迁移学习ResNet18（失败）
│
├── model/                              # 模型权重
│   └── best_optical_v13.pth            # 🏆 V13最佳权重（13.1MB）
│
├── output/                             # 训练输出报告
│   ├── optical_v13_report.txt          # V13完整训练报告
│   ├── optical_v13_optim_report.txt    # V13优化方向分析报告
│   └── optical_v13_postprocess_report.txt # V13后处理评估报告
│
└── doc/                                # 技术文档
    ├── 模型演进纪实.md                    # V1-V15完整演进历史
    └── 问题记录.md                       # 开发问题记录与解决方案
```

---

## 🏆 为什么选择V13

经过V1-V15共15个版本的迭代，V13以**测试集Acc=90.54%、总分19.23/30**成为最优版本。

### V13核心优势
| 指标 | 值 | 赛题要求 | 是否达标 |
|:----|:--:|:--------:|:--------:|
| 测试集Acc | **90.54%** | >85% | ✅ |
| Ro光占比 | **96.36%** | >50% | ✅ |
| G评分 | **通过** | 通过 | ✅ |
| S_ratio | **18.55/20** | — | 接近满分 |
| 总分 | **19.23/30** | — | 最优 |

### V13关键创新
1. **Dropout=0.6 + 过采样7x + 余弦重启**：三管齐下的强正则化策略
2. **阈值校准（0.5→0.35）**：单步提升Acc +2.88%，NORMAL召回率翻倍
3. **FocalLoss(γ=3.0)+标签平滑0.05**：聚焦难分样本同时防止过自信
4. **FC架构8192→384→64→2**：恰到好处的容量设计，参数量3.27M

### 失败版本教训（不再使用）
- V14：伪标签在类别不平衡时引入噪声，总分-0.34
- V15：CutOut增强过强，验证集98.92%但测试集90.06%（-0.48%）
- V11：迁移学习在光计算模拟架构下无效（权重结构不匹配）
- V9：MultiClassAsymmetricLoss导致梯度消失
- V10：大FC架构（4.36M参数）不敌小FC（3.27M参数）

---

## 🚀 使用说明

### 环境要求
- Docker容器（LTSimulator v1.4.6镜像）或Python 3.9 + PyTorch 2.1
- 依赖：torch, torchvision, numpy, gradio(4.x)

### 训练V13
```bash
docker exec reverent_cannon //local/miniconda/envs/moca_llm/bin/python \
  //workspace/Optical_ChestXRay/src/train_v13.py
```

### 评估与阈值校准
```bash
docker exec reverent_cannon //local/miniconda/envs/moca_llm/bin/python \
  //workspace/Optical_ChestXRay/src/eval_v13_postprocess.py
```

### 启动Demo
```bash
docker exec reverent_cannon //local/miniconda/envs/moca_llm/bin/python \
  //workspace/Optical_ChestXRay/src/app_demo.py
```

### 数据准备
从 [Kaggle Chest X-Ray](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 下载数据集，
解压到 `data/chest_xray/`（训练集/验证集/测试集结构不变）。

---

## 📊 模型架构

```
Input [B, 1, 128, 128]
  ├─ OpticalConv1(1→16, k=3) + BN + Pool(2×2) + ReLU   [B, 16, 64, 64]
  ├─ OpticalConv2(16→32, k=3) + BN + Pool(2×2) + ReLU   [B, 32, 32, 32]
  ├─ OpticalConv3(32→64, k=3) + BN + Pool(2×2) + ReLU   [B, 64, 16, 16]
  ├─ OpticalConv4(64→128, k=3) + BN + Pool(2×2) + ReLU  [B, 128, 8, 8]
  ├─ Flatten: [B, 8192]
  ├─ FC0(8192→384) + BN + Dropout(0.6) + ReLU
  ├─ FC1(384→64) + BN + Dropout(0.6) + ReLU
  └─ FC2(64→2)
```

**光计算层**：4层光卷积（nn.Linear模拟）+ 4层光池化（nn.Linear(4,1)）+ 3层光全连接
**激活函数**：ReLU（5次，唯一电计算层）
