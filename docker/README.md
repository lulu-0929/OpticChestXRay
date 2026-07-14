# Docker 容器配置说明

## 容器信息

| 项目 | 值 |
|------|-----|
| 容器名 | `reverent_cannon` |
| 镜像 | `lightelligence.docker:lt-simulator_v1.4.6-final` |
| Python | `/local/miniconda/envs/moca_llm/bin/python` |
| 环境名 | `moca_llm` |
| PyTorch | 2.1.0+cu121（CUDA 12.1） |
| torchvision | 0.16.0+cu121 |
| GPU | RTX 4070 Laptop GPU |
| 工作目录 | `/workspace/Optical_ChestXRay/` |

## 激活方式

```bash
# 交互式进入容器
docker exec -it reverent_cannon bash

# 进入后激活环境
source /local/miniconda/bin/activate moca_llm

# 直接执行单行命令
docker exec reverent_cannon /local/miniconda/envs/moca_llm/bin/python /workspace/Optical_ChestXRay/src/train_v8.py
```

## 文件复制

```bash
# 宿主机 → 容器
docker cp train_v8.py reverent_cannon:/workspace/Optical_ChestXRay/src/train_v8.py

# 容器 → 宿主机
docker cp reverent_cannon:/workspace/Optical_ChestXRay/output/best_optical_v8.pth .
```

> **注意**：Windows Git Bash 中，容器内路径前面加双斜杠 `//` 避免路径转义。
