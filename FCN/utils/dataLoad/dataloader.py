import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set='train', crop_size=(320, 320)):
        """
        :param root_dir: 你的 dataset 文件夹路径，例如 './dataset/VOCdevkit/VOC2012'
        :param image_set: 'train' (训练集) 或 'val' (验证集)
        :param crop_size: 裁剪尺寸 (H, W)。因为输入图片大小不一，为了能打包成一个 Batch，我们需要统一尺寸
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.crop_size = crop_size

        # 拼接图片和标签的文件夹路径
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.root_dir, 'SegmentationClass')

        # 读取 train.txt 或 val.txt，获取所有图片的文件名（不带后缀）
        split_f = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(split_f, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

    def __len__(self):
        """告诉 PyTorch 这个数据集一共有多少张图片"""
        return len(self.file_names)

    def _sync_transform(self, img, mask):
        # 1. 动态 Padding (填充)：如果原图比 320x320 小，我们先给它补边！
        w, h = img.size
        pad_w = max(0, self.crop_size[1] - w)
        pad_h = max(0, self.crop_size[0] - h)
        if pad_w > 0 or pad_h > 0:
            # 原图补黑色 (0)
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            # 标签图补 255 (Loss 函数里设了 ignore_index=255，所以网络不会去学这些黑边)
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)

        # 2. 随机裁剪 (现在可以保证所有图片都 >= 320x320 了，肯定能切出标准的正方形)
        i, j, h, w = self._get_crop_params(img.size)
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # 3. 随机水平翻转 (数据增强)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 4. 转 Tensor
        img = TF.to_tensor(img)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask

    def _get_crop_params(self, img_size):
        # 计算裁剪的坐标。
        w, h = img_size
        th, tw = self.crop_size
        # 因为前面做了 Pad，现在 h 和 w 绝对 >= th 和 tw
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


    def __getitem__(self, index):
        """当 PyTorch 按照索引找你要图片时，执行这里的代码"""
        file_name = self.file_names[index]

        img_path = os.path.join(self.image_dir, file_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, file_name + '.png')

        # 读取原图 (转为 RGB，以防有灰度图)
        img = Image.open(img_path).convert('RGB')
        # 读取标签图 (保持默认的 P 模式，绝不能 convert)
        mask = Image.open(mask_path)

        # 应用同步变换
        img, mask = self._sync_transform(img, mask)

        return img, mask