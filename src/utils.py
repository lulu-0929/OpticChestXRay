"""
光计算加速的医学影像辅助诊断系统 — 工具函数
===========================================
包含：日志输出、光占比计算、官方评分公式计算
"""

import time
import torch
from config import LOG_FILE, BEST_MODEL_PATH


def log_print(msg):
    """双输出函数：终端 + 文件（实时刷新）"""
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


def init_log():
    """初始化日志文件"""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("===== 曦智赛：光计算加速的医学影像辅助诊断系统 =====\n")
        f.write(f"实验时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("核心创新：用 nn.Linear 模拟卷积操作，所有核心计算均为光计算\n")
        f.write("模型：4层光模拟卷积 + 4层光池化 + 2层全连接\n")
        f.write("任务：胸部 X 光片肺炎诊断（二分类：正常/肺炎）\n")
        f.write("输入尺寸：1×128×128（灰度）\n\n")


def compute_optical_ratio(model):
    """
    计算模型的光占比 Ro
    规则：nn.Linear 层 = 光计算，ReLU 层 = 电计算
    
    返回：
        Ro: 光占比 (0~1)
        optical_ops: 光计算量
        total_ops: 总计算量
    """
    optical_ops = 0
    electrical_ops = 0
    
    log_print("\n----- 光计算层详细统计 -----")
    
    for info in model.optical_layers:
        ops = info["compute_amount"]
        optical_ops += ops
        log_print(f"  光层: {info['name']:<20} | Shape: {info['shape']:<25} | 计算量: {ops:>12,}")
    
    # 计算电计算层（仅 ReLU 激活函数）
    log_print("----- 电计算层详细统计 -----")
    
    # 模型前向传播中的 ReLU 次数：
    # ReLU1 after pool1: 输入 [batch, 16, 64, 64]
    # ReLU2 after pool2: 输入 [batch, 32, 32, 32]
    # ReLU3 after pool3: 输入 [batch, 64, 16, 16]
    # ReLU4 after pool4: 输入 [batch, 128, 8, 8]
    # ReLU5 after FC0:   输入 256 维
    
    relu_specs = [
        ("ReLU after pool1 (16×64×64)", 16 * 64 * 64),
        ("ReLU after pool2 (32×32×32)", 32 * 32 * 32),
        ("ReLU after pool3 (64×16×16)", 64 * 16 * 16),
        ("ReLU after pool4 (128×8×8)", 128 * 8 * 8),
        ("ReLU after FC0 (256)", 256),
    ]
    
    for name, ops in relu_specs:
        electrical_ops += ops
        log_print(f"  {name:<35} | 计算量: {ops:>12,}")
    
    total_ops = optical_ops + electrical_ops
    Ro = optical_ops / total_ops if total_ops > 0 else 0
    
    log_print(f"\n  光计算总量: {optical_ops:>12,}")
    log_print(f"  电计算总量: {electrical_ops:>12,}")
    log_print(f"  总计算量:   {total_ops:>12,}")
    log_print(f"  光占比 Ro:  {Ro:.6f} ({Ro*100:.2f}%)")
    
    return Ro, optical_ops, total_ops


def calculate_score(Ro, accuracy, avg_latency):
    """
    官方评分公式计算
    参考：曦智赛评分标准
    
    参数：
        Ro: 光占比 (0~1)
        accuracy: Top-1准确率
        avg_latency: 单张图片平均推理延时（秒）
    
    返回：
        G: 硬性门槛（0/1）
        S_ratio: 光占比得分 (0~20)
        S_acc: 精度得分 (0~5)
        S_lat: 延时得分 (0~5)
        total_score: 综合总分 (0~30)
    """
    # 硬性门槛：光占比 > 50% 且 准确率 > 85%
    G = 1 if (Ro > 0.5 and accuracy > 0.85) else 0
    
    # 光占比得分：S_ratio = 20 * (Ro - 0.5) / 0.5
    S_ratio = 20 * (Ro - 0.5) / 0.5
    
    # 精度得分：S_acc = 5 * ((accuracy - 0.85) / 0.15) ** 2
    S_acc = 5 * ((accuracy - 0.85) / 0.15) ** 2
    
    # 延时得分（基于初赛规则，酌情调整）
    if avg_latency > 3.6:
        S_lat = 0
    elif 0.36 < avg_latency <= 3.6:
        S_lat = 3 * (3.6 - avg_latency) / 3.24
    else:
        S_lat = 3 + 2 * (0.36 - avg_latency) / 0.36
    
    total_score = G * (S_ratio + S_acc + S_lat)
    return G, S_ratio, S_acc, S_lat, total_score


def print_final_report(Ro, accuracy, total_latency, avg_latency, 
                       G, S_ratio, S_acc, S_lat, total_score):
    """输出最终评估报告"""
    log_print("\n" + "=" * 70)
    log_print("【曦智赛：光计算加速的医学影像辅助诊断系统 — 最终评估】")
    log_print("=" * 70)
    log_print(f"光占比 Ro:           {Ro:.4f} ({Ro*100:.2f}%)")
    log_print(f"分类精度 Acc:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_print(f"测试集推理总延时:     {total_latency:.4f} s")
    log_print(f"单张图片推理延时:     {avg_latency*1000:.4f} ms")
    log_print("")
    log_print("----- 官方评分结果 -----")
    log_print(f"硬性门槛 G:          {G} ({'通过' if G==1 else '未通过'})")
    log_print(f"光占比得分 S_ratio:  {S_ratio:.2f} / 20.00")
    log_print(f"精度得分 S_acc:      {S_acc:.2f} / 5.00")
    log_print(f"延时得分 S_lat:      {S_lat:.2f} / 5.00")
    log_print(f"综合总分 S:          {total_score:.2f} / 30.00")
    log_print("=" * 70)


def print_optical_layers(model, optical_ops, Ro):
    """输出光计算层详细清单"""
    log_print("\n" + "=" * 70)
    log_print("【光计算层详细清单】")
    log_print("=" * 70)
    log_print("说明：以下所有 Linear 层均由光计算调度")
    log_print(f"{'序号':<4} {'层名称':<22} {'Shape':<30} {'计算量':<15}")
    log_print("-" * 80)
    
    for idx, info in enumerate(model.optical_layers, 1):
        log_print(f"{idx:<4} {info['name']:<22} {info['shape']:<30} {info['compute_amount']:<15,}")
    
    log_print("-" * 80)
    log_print(f"{'总计':<4} {'':<22} {'':<30} {optical_ops:<15,}")
    log_print(f"光占比 Ro = {Ro*100:.2f}%")
