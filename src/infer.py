"""
光计算加速的医学影像辅助诊断系统 — 推理与评分脚本
================================================
加载训练好的最佳模型，在测试集上进行推理评估
计算光占比、评分公式，输出最终报告
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from model import OpticalChestXRay
from utils import (
    log_print, init_log, compute_optical_ratio, 
    calculate_score, print_final_report, print_optical_layers
)


def get_test_loader(data_root):
    """加载测试集"""
    from torchvision import datasets
    
    test_path = os.path.join(data_root, "chest_xray", "test")
    
    test_dataset = datasets.ImageFolder(
        root=test_path, 
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    log_print(f"\n测试集加载完成:")
    log_print(f"  测试集: {len(test_dataset)} 张")
    log_print(f"  类别: {test_dataset.classes}")
    
    return test_loader, test_dataset.classes


def main():
    init_log()
    
    log_print("===== 加载最佳模型进行最终评估 =====")
    log_print(f"设备: {device}")
    log_print(f"输入尺寸: {IN_CHANNELS}×{IMG_SIZE}×{IMG_SIZE}")
    log_print(f"任务: 胸部 X 光肺炎诊断（二分类）")
    
    # 加载测试集
    test_loader, classes = get_test_loader(DATA_PATH)
    
    # 加载模型（推理模式，dropout_rate=0）
    model = OpticalChestXRay(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, dropout_rate=0.0).to(device)
    
    if os.path.exists(BEST_MODEL_PATH):
        state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
        # 使用strict=False加载（dropout层无参数，不影响权重加载）
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            log_print(f"加载提示（可忽略）: {missing_keys}")
        log_print(f"已加载预训练模型: {BEST_MODEL_PATH}")
    else:
        log_print("警告: 未找到预训练模型，使用未训练的模型进行评估")
    
    model.eval()
    torch.set_grad_enabled(False)
    
    # ===== 推理 =====
    log_print("\n----- 开始推理 -----")
    inference_start = time.time()
    
    correct = 0
    total = 0
    
    # 用于计算各类别准确率
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 逐类别统计
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    inference_end = time.time()
    total_latency = inference_end - inference_start
    avg_latency = total_latency / len(test_loader.dataset)
    accuracy = correct / total
    
    log_print(f"推理完成: {total} 张图片, 耗时 {total_latency:.4f}s")
    log_print(f"平均每张: {avg_latency*1000:.4f}ms")
    
    # 各类别准确率
    log_print("\n----- 各类别诊断准确率 -----")
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            log_print(f"  {class_name:<12}: {class_correct[i]:>4d}/{class_total[i]:>4d} ({class_acc*100:.2f}%)")
    
    # ===== 计算光占比 =====
    Ro, optical_ops, total_ops = compute_optical_ratio(model)
    
    # ===== 官方评分 =====
    G, S_ratio, S_acc, S_lat, total_score = calculate_score(Ro, accuracy, avg_latency)
    
    # ===== 输出最终报告 =====
    print_final_report(Ro, accuracy, total_latency, avg_latency, 
                       G, S_ratio, S_acc, S_lat, total_score)
    
    # ===== 光计算层详细报告 =====
    print_optical_layers(model, optical_ops, Ro)
    
    log_print("\n===== 评估完成 =====")
    
    return {
        "accuracy": accuracy,
        "Ro": Ro,
        "total_latency": total_latency,
        "avg_latency": avg_latency,
        "G": G,
        "S_ratio": S_ratio,
        "S_acc": S_acc,
        "S_lat": S_lat,
        "total_score": total_score,
    }


if __name__ == "__main__":
    results = main()
