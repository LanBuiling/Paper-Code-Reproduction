# tests/UNet/unet_smp_tester.py
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp


class SMPTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 1. 初始化模型并加载权重
        # 测试时不需要下载预训练权重，因为我们要加载自己微调好的
        self.model = smp.Unet(
            encoder_name="vgg16_bn",
            in_channels=3,
            classes=21
        ).to(self.device)

        # 指向我们刚训练好的 VGG 混血权重
        checkpoint_path = os.path.join('checkpoints', 'UNet_SMP', 'unet_vgg16_best.pth')

        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"✅ 成功加载工业级权重: {checkpoint_path}")
        else:
            print("❌ 警告：未找到权重文件！")

        self.model.eval()
        self.palette = self._get_voc_palette()

        # 2. 现代版测试数据预处理：必须加 ImageNet 归一化！
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_voc_palette(self):
        palette = torch.tensor([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
            [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ], dtype=torch.uint8)
        return palette

    def test(self):
        # 创建新的结果保存目录，防混淆
        save_dir = os.path.join('results', 'UNet_SMP')
        os.makedirs(save_dir, exist_ok=True)

        test_dir = os.path.join(self.args.data_path, 'JPEGImages')
        sample_names = os.listdir(test_dir)[:5]

        print(f"正在开始测试，叠加结果将保存至: {save_dir}")

        with torch.no_grad():
            for name in sample_names:
                img_path = os.path.join(test_dir, name)
                raw_img = Image.open(img_path).convert('RGB')

                # 3. 准备两份图片：
                # 份 A (img_tensor)：经过归一化，专门用来喂给网络做预测
                img_tensor = self.img_transform(raw_img).unsqueeze(0).to(self.device)

                # 份 B (img_for_show)：不经过归一化，保留原始色彩，专门用来做透明度叠加
                img_for_show = raw_img.resize((512, 512), resample=Image.BILINEAR)

                # 推理：得到 [1, 21, 512, 512]
                output = self.model(img_tensor)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu()

                # 染色可视化
                color_mask = self.palette[pred]
                color_mask_img = Image.fromarray(color_mask.numpy()).convert('RGB')

                # ==========================================
                # 核心改变：不再需要复杂的中心裁剪，直接 1:1 叠加！
                # 因为 SMP 里的 Padding 是 Same 模式，输入 512 输出就是 512
                # ==========================================
                alpha_blend = 0.6
                blended_img = Image.blend(img_for_show, color_mask_img, alpha_blend)

                blended_img.save(os.path.join(save_dir, f"smp_res_{name.split('.')[0]}.png"))
                print(f"已处理并保存: smp_res_{name.split('.')[0]}.png")