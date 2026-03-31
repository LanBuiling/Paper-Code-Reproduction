import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataLoad.dataloader import VOCSegmentationDataset


# =========================================================
# 魔法 1：手工打造原作者的双线性插值卷积核 (Bilinear Kernel)
# =========================================================
def bilinear_kernel(in_channels, out_channels, kernel_size):
    """根据数学公式生成双线性插值的权重矩阵"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


# =========================================================
# 魔法 2：严格按照论文 4.3 节初始化模型
# =========================================================
def apply_paper_initialization(model):
    for m in model.modules():
        # 1. 如果是反卷积层，注入双线性插值权重
        if isinstance(m, nn.ConvTranspose2d):
            m.weight.data.copy_(bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 2. 如果是最后打分的 1x1 卷积层 (输出21类的那层)，全零初始化
        elif isinstance(m, nn.Conv2d) and m.kernel_size[0] == 1 and m.out_channels == 21:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# =========================================================
# 核心训练逻辑
# =========================================================
def train_model(args, model, device):
    print(f"🔥 开始执行训练任务: 模型={args.model}, 最大Epochs={args.epochs}")

    # 注入原著的 [全零打分层] 与 [双线性反卷积] 初始化
    apply_paper_initialization(model)

    # 准备数据
    train_dataset = VOCSegmentationDataset(root_dir='./dataset/VOCdevkit/VOC2012', image_set='train',
                                           crop_size=(320, 320))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = VOCSegmentationDataset(root_dir='./dataset/VOCdevkit/VOC2012', image_set='val', crop_size=(320, 320))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 【回归原著】：使用原作者设定的 SGD 参数
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    # 早停策略相关变量
    best_val_loss = float('inf')
    patience = 5  # 5 轮验证集平均 loss 不降
    trigger_times = 0

    for epoch in range(args.epochs):
        # ---------------------------------------------------------
        # 阶段 1：每个 Batch 计算 Loss 并反向传播更新参数
        # ---------------------------------------------------------
        model.train()
        total_train_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], Batch [{i + 1}/{len(train_loader)}], 当前 Batch Loss: {loss.item():.4f}")

        # 计算这一轮的【平均训练 Loss】
        avg_train_loss = total_train_loss / len(train_loader)

        # ---------------------------------------------------------
        # 阶段 2：每轮结束后计算【平均验证 Loss】
        # ---------------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"\n📊 Epoch {epoch + 1} 总结 | 平均训练 Loss: {avg_train_loss:.4f} | 平均验证 Loss: {avg_val_loss:.4f}")

        # ---------------------------------------------------------
        # 阶段 3：早停策略 (Early Stopping) 判断
        # ---------------------------------------------------------
        if avg_val_loss < best_val_loss:
            print(f"🎉 验证集平均 Loss 创历史新低 ({best_val_loss:.4f} -> {avg_val_loss:.4f})！")
            best_val_loss = avg_val_loss
            trigger_times = 0

            save_path = os.path.join('checkpoints', f'{args.model}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"💾 最佳模型权重已保存至: {save_path}")
        else:
            trigger_times += 1
            print(
                f"⚠️ 验证集平均 Loss 未下降 ({avg_val_loss:.4f} >= {best_val_loss:.4f})。早停警告: {trigger_times}/{patience}")

            if trigger_times >= patience:
                print("🛑 已达到最大容忍度，触发早停策略 (Early Stopping)！提前终止训练。")
                break
        print("-" * 60)