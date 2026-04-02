# tests/UNet/unet_tester.py
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from models.Unet_original import UNetOriginal


class UNetTester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 1. 初始化模型并加载权重
        self.model = UNetOriginal(n_channels=3, n_classes=21).to(self.device)
        checkpoint_path = os.path.join('checkpoints', 'UNet', 'unet_original_best.pth')  # 假设你想测第50轮
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"成功加载权重: {checkpoint_path}")
        else:
            print("警告：未找到权重文件，将使用随机初始化参数进行测试！")

        self.model.eval()

        # 2. VOC2012 调色板 (用于把 0-20 的索引转回彩色图)
        self.palette = self._get_voc_palette()

    def _get_voc_palette(self):
        # 标准 VOC 调色板，用于可视化
        palette = torch.tensor([
            [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
            [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128]
        ], dtype=torch.uint8)
        return palette

    def test(self):
        # 创建结果保存目录
        save_dir = os.path.join('results', 'UNet')
        os.makedirs(save_dir, exist_ok=True)

        # 3. 寻找测试图片 (从验证集里随机挑几张试试)
        test_dir = os.path.join(self.args.data_path, 'JPEGImages')
        # 假设我们只测前 5 张
        sample_names = os.listdir(test_dir)[:10]

        print(f"正在开始测试，结果将保存至: {save_dir}")

        with torch.no_grad():
            for name in sample_names:
                img_path = os.path.join(test_dir, name)
                raw_img = Image.open(img_path).convert('RGB')

                # 原版 U-Net 必须输入 572x572
                img = raw_img.resize((572, 572), resample=Image.BILINEAR)
                img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

                # 推理：得到 [1, 21, 388, 388]
                output = self.model(img_tensor)

                # 取概率最大的类别索引: [388, 388]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu()

                # 4. 染色可视化
                # 将索引图转为 RGB 图
                color_mask = self.palette[pred]  # [388, 388, 3]
                color_mask_img = Image.fromarray(color_mask.numpy())

                # ==========================================
                # --- 核心修改在此 ---
                # ==========================================

                # 5. 获取模型真正对应的输入区域：从 572x572 的输入图中，裁剪出中心的 388x388 区域。
                # 由于 Valid Padding，预测图只对应输入图的中心部分。
                # PIL Image crop(left, top, right, bottom). 中心裁剪偏移量为 (572-388)/2 = 92
                img_cropped = img.crop((92, 92, 92 + 388, 92 + 388))

                # 6. 将裁剪区域与分割图进行透明度叠加
                # 确保两者都是 RGB 模式以供混合
                img_cropped = img_cropped.convert('RGB')
                color_mask_img = color_mask_img.convert('RGB')

                # Image.blend(background, overlay, alpha)
                # alpha 控制 overlay 的透明度：1.0 为掩码不透明，0.0 为完全透明（显示原图）
                alpha_blend = 0.6  # 设置掩码的 opacity 为 60%，可调
                blended_img = Image.blend(img_cropped, color_mask_img, alpha_blend)

                # 7. 保存叠加结果
                # 修改文件名以区分：blended_res_...
                blended_img.save(os.path.join(save_dir, f"blended_res_{name.split('.')[0]}.png"))
                print(f"已处理并保存叠加结果: blended_res_{name.split('.')[0]}.png")