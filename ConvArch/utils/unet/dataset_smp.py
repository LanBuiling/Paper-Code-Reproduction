# utils/UNet/dataset_smp.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VOCSegDatasetSMP(Dataset):
    def __init__(self, root_dir, image_set='train'):
        super().__init__()
        self.root_dir = root_dir

        image_dir = os.path.join(root_dir, 'JPEGImages')
        mask_dir = os.path.join(root_dir, 'SegmentationClass')

        split_f = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

        # 核心改动：缩放至 512x512，并执行 ImageNet 标准归一化
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        # 1. 预处理图片
        img_tensor = self.img_transform(img)

        # 2. 预处理掩码 (只缩放尺寸，千万不能做归一化，保持 0-20 的类别索引)
        mask = mask.resize((512, 512), resample=Image.NEAREST)
        mask_np = np.array(mask)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)

        # 现代 U-Net 直接用交叉熵，不需要返回权重图了
        return img_tensor, mask_tensor