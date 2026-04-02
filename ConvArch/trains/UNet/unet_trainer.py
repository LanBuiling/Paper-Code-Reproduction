# trains/UNet/unet_trainer.py
import torch
import os
from torch.utils.data import DataLoader
from models.Unet_original import UNetOriginal
from utils.unet.dataset_original import VOCSegDatasetOriginal
from utils.unet.loss import UNetWeightedLoss


class UNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.model = UNetOriginal(n_channels=3, n_classes=21).to(self.device)

        # 1. 分别加载训练集和验证集
        self.train_ds = VOCSegDatasetOriginal(args.data_path, image_set='train')
        self.val_ds = VOCSegDatasetOriginal(args.data_path, image_set='val')  # 加载验证集

        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False)

        self.criterion = UNetWeightedLoss(ignore_index=255).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # 2. 早停与保存相关状态
        self.best_loss = float('inf')  # 初始设为无穷大
        self.early_counter = 0  # 累计没有改善的次数
        self.ckpt_dir = os.path.join('checkpoints', 'UNet')
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def train(self):
        print(f"开始训练... 目标模型: {self.args.model} , 早停轮数: {self.args.early}")

        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = 0.0

            # 【修复核心】：加回 enumerate 来追踪具体的 Batch (Step)
            for step, (imgs, masks, weights) in enumerate(self.train_loader):
                imgs, masks, weights = imgs.to(self.device), masks.to(self.device), weights.to(self.device)

                # 前向传播与计算 Loss
                preds = self.model(imgs)
                loss = self.criterion(preds, masks, weights)

                # 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

                # 【心跳监测】：每跑 10 个 Batch，就打印一次当前的实时 Loss
                if step % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.args.epochs}] Step [{step}/{len(self.train_loader)}] | Batch Loss: {loss.item():.4f}")

            # 加入早停机制
            val_loss = self.validate()
            avg_train_loss = train_loss / len(self.train_loader)

            # 打印整个 Epoch 的总结报告
            print(
                f"\n>>> Epoch {epoch + 1} : 平均 Train Loss: {avg_train_loss:.4f} , 平均 Val Loss: {val_loss:.4f}\n")

            # 早停与保存逻辑
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.early_counter = 0
                save_path = os.path.join(self.ckpt_dir, f"{self.args.model}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"最佳模型已保存至: {save_path}\n")
            else:
                self.early_counter += 1
                print(f"触发早停，early: {self.early_counter}/{self.args.early}\n")

                if self.early_counter >= self.args.early:
                    print("触发早停：验证集 Loss 已长时间不再下降，停止训练以防过拟合。")
                    break
    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks, weights in self.val_loader:
                imgs, masks, weights = imgs.to(self.device), masks.to(self.device), weights.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks, weights)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)