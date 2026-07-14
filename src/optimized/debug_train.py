import sys, os, time, torch, torch.nn as nn, random, numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
sys.path.insert(0, "/workspace/Optical_ChestXRay/src")
from train_v9 import OpticalChestXRayV9, MultiClassAsymmetricLoss, WeightedSubsetDataset, normal_strong_transform, pneumonia_transform, mixup_data

device = torch.device("cuda")
DATA_PATH = "/workspace/Optical_ChestXRay/data"

# 构建完整数据集
train_full = datasets.ImageFolder(os.path.join(DATA_PATH, "chest_xray", "train"), transform=None)
all_labels = train_full.targets
class_names = train_full.classes
normal_indices = [i for i, lbl in enumerate(all_labels) if lbl == 0]
pneumonia_indices = [i for i, lbl in enumerate(all_labels) if lbl == 1]
random.seed(42)
val_normal = random.sample(normal_indices, int(0.25 * len(normal_indices)))
val_pneumonia = random.sample(pneumonia_indices, int(0.065 * len(pneumonia_indices)))
val_indices = val_normal + val_pneumonia
random.shuffle(val_indices)
train_indices = list(set(range(len(train_full))) - set(val_indices))
train_normal = [i for i in train_indices if all_labels[i] == 0]
train_pneumonia = [i for i in train_indices if all_labels[i] == 1]
oversampled = train_normal * 10 + train_pneumonia
random.shuffle(oversampled)

print(f"Train: NORMAL={len(train_normal)*10}, PNEUMONIA={len(train_pneumonia)}")
print(f"Val: NORMAL={len(val_normal)}, PNEUMONIA={len(val_pneumonia)}")

# 构建训练Dataloader（使用WeightedSubsetDataset）
train_ds = WeightedSubsetDataset(train_full, oversampled, [normal_strong_transform, pneumonia_transform])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

# 验证集
val_full = datasets.ImageFolder(os.path.join(DATA_PATH, "chest_xray", "train"),
    transform=transforms.Compose([
        transforms.Resize((128, 128)), transforms.Grayscale(),
        transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]),
    ]))
val_ds = Subset(val_full, val_indices)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

model = OpticalChestXRayV9(dropout_rate=0.5).to(device)
criterion = MultiClassAsymmetricLoss(gamma_pos=0.0, gamma_neg=2.0, smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=8e-4)

# 诊断：先看第1个batch的输出
model.train()
for bidx, (img, lbl) in enumerate(train_loader):
    if bidx > 0: break
    img, lbl = img.to(device), lbl.to(device)
    print(f"Batch {bidx}: img.shape={img.shape}, img.min={img.min().item():.3f}, img.max={img.max().item():.3f}")
    print(f"Labels dist: NORMAL={(lbl==0).sum().item()}, PNEUMONIA={(lbl==1).sum().item()}")
    
    outputs = model(img)
    print(f"Outputs: mean={outputs.mean().item():.3f}, std={outputs.std().item():.3f}")
    probs = torch.softmax(outputs, dim=1)
    _, preds = torch.max(outputs, 1)
    correct = (preds == lbl).sum().item()
    print(f"Init pred: NORMAL={(preds==0).sum().item()}, PNEUMONIA={(preds==1).sum().item()}, Acc={correct/64*100:.1f}%")
    print(f"Init probs[:5]: {probs[:5].tolist()}")
    
    # 第一次前向+反向
    loss = criterion(outputs, lbl)
    print(f"Init loss: {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    print(f"Grad norm: {grad_norm:.4f}")
    print(f"fc2.bias.grad: {model.fc2.bias.grad.tolist()}")
    
    # 更新一步
    optimizer.step()
    
    # 再预测一次
    outputs2 = model(img)
    _, preds2 = torch.max(outputs2, 1)
    correct2 = (preds2 == lbl).sum().item()
    print(f"After 1 step pred: NORMAL={(preds2==0).sum().item()}, PNEUMONIA={(preds2==1).sum().item()}, Acc={correct2/64*100:.1f}%")
    break

# 完整训练10个epoch，每epoch验证
print("\n===== 10个epoch快速测试 =====")
ce_epochs = 5
for epoch in range(ce_epochs):
    model.train()
    total_loss = 0
    for bidx, (img, lbl) in enumerate(train_loader):
        img, lbl = img.to(device), lbl.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    # 验证
    model.eval()
    correct = total = 0
    ccorrect, ctotal = [0,0], [0,0]
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), lbl.to(device)
            outputs = model(img) / 1.0
            _, pred = torch.max(outputs, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
            for i in range(lbl.size(0)):
                lb = lbl[i].item()
                ctotal[lb] += 1
                if pred[i].item() == lb:
                    ccorrect[lb] += 1
    
    acc = correct/total
    n_recall = ccorrect[0]/max(ctotal[0],1)
    p_recall = ccorrect[1]/max(ctotal[1],1)
    print(f"E{epoch+1}/{ce_epochs} Loss={total_loss/len(train_loader):.4f} ValAcc={acc*100:.2f}% N={n_recall*100:.1f}% P={p_recall*100:.1f}%")
    
    if n_recall > 0.1:
        print("  NORMAL开始被正确分类! 训练方向正确!")
