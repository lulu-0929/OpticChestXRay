"""
光计算加速的医学影像辅助诊断系统 — 训练脚本 V5（优化版）
改进：类别平衡 + 加权损失 + 训练集内分割验证集
"""
import time, torch, torch.nn as nn, torch.optim as optim, os, sys, numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms

IMG_SIZE=128; IN_CHANNELS=1; NUM_CLASSES=2; batch_size=64; epochs=30; lr=1e-3; weight_decay=1e-4
BASE_DIR='/workspace/Optical_ChestXRay'; DATA_PATH=os.path.join(BASE_DIR,'data')
LOG_FILE=os.path.join(BASE_DIR,'output','optical_v5_report.txt')
BEST_MODEL_PATH=os.path.join(BASE_DIR,'output','best_optical_v5.pth')
os.makedirs(os.path.dirname(LOG_FILE),exist_ok=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log_print(msg):
    print(msg)
    with open(LOG_FILE,"a",encoding="utf-8") as f: f.write(msg+"\n")

def init_log():
    with open(LOG_FILE,"w",encoding="utf-8") as f:
        f.write("===== 光计算加速医学影像诊断 V5 =====\n")
        f.write(f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.Grayscale(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5,translate=(0.05,0.05)),
    transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.Grayscale(),
    transforms.ToTensor(), transforms.Normalize(mean=[0.5],std=[0.5])])

sys.path.insert(0,os.path.join(BASE_DIR,'src'))
from model import OpticalChestXRay

def main():
    init_log()
    log_print(f"设备: {device} | lr={lr} | epochs={epochs} | batch={batch_size}")
    
    train_full = datasets.ImageFolder(root=os.path.join(DATA_PATH,'chest_xray','train'),transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(DATA_PATH,'chest_xray','test'),transform=test_transform)
    
    val_size=int(0.15*len(train_full)); train_size=len(train_full)-val_size
    train_dataset,val_dataset = random_split(train_full,[train_size,val_size],generator=torch.Generator().manual_seed(42))
    
    train_labels=[train_full.targets[i] for i in train_dataset.indices]
    class_counts=np.bincount(train_labels)
    log_print(f"训练集分布: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}")
    
    sample_weights = 1.0/class_counts[train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights,num_samples=len(sample_weights),replacement=True)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,sampler=sampler,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    log_print(f"训练: {train_size}(加权) | 验证: {val_size} | 测试: {len(test_dataset)}")
    
    model=OpticalChestXRay(in_channels=IN_CHANNELS,num_classes=NUM_CLASSES).to(device)
    
    # 类别加权损失
    class_weight_tensor=torch.tensor([1.0/1341,1.0/3875],dtype=torch.float32).to(device)
    class_weight_tensor=class_weight_tensor/class_weight_tensor.sum()*NUM_CLASSES
    criterion=nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    
    log_print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    best_acc=0.0; start=time.time()
    for epoch in range(epochs):
        model.train(); torch.set_grad_enabled(True); total_loss=0.0
        for bidx,(img,lbl) in enumerate(train_loader):
            img,lbl=img.to(device),lbl.to(device)
            optimizer.zero_grad(); outputs=model(img)
            loss=criterion(outputs,lbl); loss.backward(); optimizer.step()
            total_loss+=loss.item()
            if bidx%20==0: print(f"  E{epoch+1:2d}/{epochs} B{bidx:3d} Loss:{loss.item():.4f}")
        scheduler.step()
        
        model.eval(); torch.set_grad_enabled(False)
        correct=total=0; vloss=0; cc=[0,0]; ct=[0,0]
        with torch.no_grad():
            for img,lbl in val_loader:
                img,lbl=img.to(device),lbl.to(device); outputs=model(img)
                vloss+=criterion(outputs,lbl).item()
                _,pred=torch.max(outputs,1); total+=lbl.size(0)
                correct+=(pred==lbl).sum().item()
                for i in range(lbl.size(0)):
                    lb=lbl[i].item(); ct[lb]+=1
                    if pred[i].item()==lb: cc[lb]+=1
        acc=correct/total
        cls_str=' | '.join([f"{train_full.classes[i]}: {cc[i]}/{ct[i]}({cc[i]/max(ct[i],1)*100:.1f}%)" for i in range(2)])
        log_print(f"E[{epoch+1:2d}/{epochs}] L:{total_loss/len(train_loader):.4f} VL:{vloss/len(val_loader):.4f} Acc:{acc:.4f}({acc*100:.2f}%) [{cls_str}] T:{time.time()-start:.0f}s")
        if acc>best_acc:
            best_acc=acc; torch.save(model.state_dict(),BEST_MODEL_PATH)
            log_print(f"  -> 新最佳 {best_acc*100:.2f}%")
    
    log_print(f"\n训练完成! {time.time()-start:.0f}s 最佳: {best_acc*100:.2f}%")
    
    # 测试
    model.load_state_dict(torch.load(BEST_MODEL_PATH)); model.eval()
    tc=tt=0; tcc=[0,0]; tct=[0,0]
    with torch.no_grad():
        for img,lbl in test_loader:
            img,lbl=img.to(device),lbl.to(device); outputs=model(img)
            _,pred=torch.max(outputs,1); tt+=lbl.size(0)
            tc+=(pred==lbl).sum().item()
            for i in range(lbl.size(0)):
                lb=lbl[i].item(); tct[lb]+=1
                if pred[i].item()==lb: tcc[lb]+=1
    test_acc=tc/tt
    log_print(f"测试集: {test_acc*100:.2f}%")
    for i in range(2): log_print(f"  {train_full.classes[i]}: {tcc[i]}/{tct[i]}({tcc[i]/max(tct[i],1)*100:.1f}%)")
    
    # 光占比计算
    log_print("\n----- 光占比计算 -----")
    optical_ops=sum(info['compute_amount'] for info in model.optical_layers)
    electrical_ops=16*64*64+32*32*32+64*16*16+128*8*8+256
    total_ops=optical_ops+electrical_ops; Ro=optical_ops/total_ops
    log_print(f"光计算: {optical_ops:,} | 电计算: {electrical_ops:,} | Ro={Ro*100:.2f}%")
    
    G=1 if(Ro>0.5 and test_acc>0.85) else 0
    S_ratio=20*(Ro-0.5)/0.5
    S_acc=5*((test_acc-0.85)/0.15)**2
    avg_latency=0.0  # placeholder
    log_print(f"G={G} | S_ratio={S_ratio:.2f}/20 | S_acc={S_acc:.2f}/5")

if __name__=="__main__":
    main()
