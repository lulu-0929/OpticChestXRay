"""
光计算加速的医学影像辅助诊断系统 — 训练脚本
===========================================
基于 LTSimulator 光子计算模拟器平台
训练胸部 X 光肺炎诊断模型
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from model import OpticalChestXRay
from utils import log_print, init_log


def get_data_loaders(data_root):
    """
    加载 Chest X-Ray 数据集
    
    预期目录结构：
    data/
      chest_xray/
        train/
          NORMAL/
          PNEUMONIA/
        test/
          NORMAL/
          PNEUMONIA/
    
    验证集从训练集划分 10%
    """
    from torchvision import datasets
    from torch.utils.data import random_split, Subset
    import numpy as np
    
    train_path = os.path.join(data_root, "chest_xray", "train")
    test_path = os.path.join(data_root, "chest_xray", "test")
    
    # 加载完整训练集
    full_train_dataset = datasets.ImageFolder(
        root=train_path, 
        transform=train_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=test_path, 
        transform=test_transform
    )
    
    # 从训练集划分 10% 作为验证集（保持类别分布）
    from torch.utils.data import WeightedRandomSampler
    targets = torch.tensor(full_train_dataset.targets)
    class_counts = torch.bincount(targets)
    
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    # 分层划分，保持类别比例
    train_indices, val_indices = [], []
    for cls in range(len(full_train_dataset.classes)):
        cls_indices = torch.where(targets == cls)[0].tolist()
        n_val = max(1, int(0.1 * len(cls_indices)))
        n_train = len(cls_indices) - n_val
        # 打乱并划分
        import random
        random.Random(42).shuffle(cls_indices)
        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    # 对验证集使用 test_transform（无增强）
    val_dataset.dataset.transform = test_transform
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    log_print(f"\n数据集加载完成:")
    log_print(f"  训练集: {len(train_dataset)} 张 (从训练集划分 90%)")
    log_print(f"  验证集: {len(val_dataset)} 张 (从训练集划分 10%)")
    log_print(f"  测试集: {len(test_dataset)} 张 (原始测试集)")
    log_print(f"  类别: {full_train_dataset.classes}")
    
    return train_loader, val_loader, test_loader, full_train_dataset.classes


def main():
    init_log()
    
    log_print("===== 开始训练：光计算加速的胸部 X 光肺炎诊断系统 =====")
    log_print(f"设备: {device}")
    log_print(f"输入尺寸: {IN_CHANNELS}×{IMG_SIZE}×{IMG_SIZE}")
    log_print(f"类别数: {NUM_CLASSES}")
    log_print(f"Batch Size: {batch_size}")
    log_print(f"Epochs: {epochs}")
    log_print(f"初始学习率: {lr}")
    log_print(f"Weight Decay: {weight_decay}")
    log_print(f"Dropout Rate: {dropout_rate}")
    
    # 加载数据
    train_loader, val_loader, test_loader, classes = get_data_loaders(DATA_PATH)
    
    # 初始化模型
    model = OpticalChestXRay(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dropout_rate=dropout_rate).to(device)
    
    # 使用标准损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    total_params = sum(p.numel() for p in model.parameters())
    log_print(f"\n模型参数量: {total_params:,}")
    log_print(f"光计算层数量: {len(model.optical_layers)}")
    log_print(f"光计算层清单:")
    for info in model.optical_layers:
        log_print(f"  {info['name']:<22} Shape: {info['shape']:<25} 计算量: {info['compute_amount']:>10,}")
    
    # 训练循环
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        # === 训练阶段 ===
        model.train()
        torch.set_grad_enabled(True)
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs} | Batch {batch_idx:3d}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # === 验证阶段 ===
        model.eval()
        torch.set_grad_enabled(False)
        
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        elapsed = time.time() - start_time
        log_print(
            f"Epoch [{epoch+1:2d}/{epochs}] | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Acc: {acc:.4f} ({acc*100:.2f}%) | "
            f"Time: {elapsed:.1f}s"
        )
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            log_print(f"  → 新最佳模型已保存 (Acc: {best_acc*100:.2f}%)")
    
    total_time = time.time() - start_time
    log_print(f"\n===== 训练完成！总耗时: {total_time:.1f}s =====")
    log_print(f"最佳验证准确率: {best_acc*100:.2f}%")
    
    # 在测试集上评估
    log_print(f"\n===== 在测试集上评估最佳模型 =====")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    torch.set_grad_enabled(False)
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_acc = test_correct / test_total
    log_print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return model


if __name__ == "__main__":
    main()
