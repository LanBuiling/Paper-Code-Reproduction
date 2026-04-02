# trains/UNet/unet_smp_trainer.py
import torch
import os
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.unet.dataset_smp import VOCSegDatasetSMP


class SMPTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # ==========================================
        # 魔法所在：一行代码召唤 VGG-16 加持的 U-Net
        # ==========================================
        self.model = smp.Unet(
            encoder_name="vgg16_bn",  # 骨干网络选用带 Batch Norm 的 VGG16，收敛更快
            encoder_weights="imagenet",  # 自动下载并注入 ImageNet 预训练权重
            in_channels=3,
            classes=21,
        ).to(self.device)

        self.train_ds = VOCSegDatasetSMP(args.data_path, image_set='train')
        self.val_ds = VOCSegDatasetSMP(args.data_path, image_set='val')

        self.train_loader = DataLoader(self.train_ds, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=args.batch_size, shuffle=False)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.ckpt_dir = os.path.join('checkpoints', 'UNet_SMP')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.best_loss = float('inf')

    def train(self):
        print(f"🚀 开始微调训练... 骨干网络: VGG-16")

        for epoch in range(self.args.epochs):
            self.model.train()
            train_loss = 0.0

            for step, (imgs, masks) in enumerate(self.train_loader):
                imgs, masks = imgs.to(self.device), masks.to(self.device)

                preds = self.model(imgs)
                loss = self.criterion(preds, masks)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                if step % 10 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.args.epochs}] Step [{step}/{len(self.train_loader)}] | Loss: {loss.item():.4f}")

            val_loss = self.validate()
            print(
                f">>> Epoch {epoch + 1} 结束 | Train Loss: {train_loss / len(self.train_loader):.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                save_path = os.path.join(self.ckpt_dir, "unet_vgg16_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"🎉 最佳模型已保存至: {save_path}\n")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in self.val_loader:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                preds = self.model(imgs)
                loss = self.criterion(preds, masks)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)