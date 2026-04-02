# utils/UNet/dataset_original.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import distance_transform_edt


class VOCSegDatasetOriginal(Dataset):
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

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        # 1. 缩放与裁剪 (完全遵循原版 U-Net 尺寸要求)
        img = img.resize((572, 572), resample=Image.BILINEAR)
        mask = mask.resize((572, 572), resample=Image.NEAREST)
        mask_cropped = transforms.CenterCrop(388)(mask)

        # 2. 转换为 Tensor 和 Numpy
        img_tensor = transforms.ToTensor()(img)
        mask_np = np.array(mask_cropped)
        mask_tensor = torch.as_tensor(mask_np, dtype=torch.long)

        # =====================================================================
        # 【原版论文真实做法 (理论锚点，仅供学习参考)】
        # 适用场景：实例分割（如 ISBI 细胞追踪赛，能区分 细胞A 和 细胞B）
        # 核心公式：w(x) = wc(x) + w0 * exp(-(d1(x) + d2(x))^2 / (2*sigma^2))
        # =====================================================================
        """
        # 假设 mask_np 中不同的数字代表不同的细胞个体 (1, 2, 3...)，0是背景
        instances = np.unique(mask_np)[1:] # 提取所有细胞的ID，排除背景0

        if len(instances) > 1: # 至少有两个细胞才能算 d1 和 d2
            # 建立一个 3D 矩阵，存放每个像素到各个独立细胞边缘的距离
            distances = np.zeros((len(instances), mask_np.shape[0], mask_np.shape[1]))

            for i, inst_id in enumerate(instances):
                # 提取单个细胞的二值掩码
                single_cell_mask = (mask_np == inst_id)
                # 计算全图所有像素到这个细胞边缘的绝对距离
                distances[i] = distance_transform_edt(1 - single_cell_mask)

            # 沿着细胞通道维度排序，找出最近(d1)和第二近(d2)的距离
            distances.sort(axis=0)
            d1 = distances[0] # 到最近细胞的距离
            d2 = distances[1] # 到第二近细胞的距离

            w0 = 10.0
            sigma = 5.0
            # 严格套用原论文惩罚项公式
            penalty = w0 * np.exp(- ((d1 + d2) ** 2) / (2 * sigma ** 2))

            # 注意：原作者通常只对处于背景(缝隙)的像素施加这种极强的分割惩罚
            weight_np = np.ones_like(mask_np, dtype=np.float32)
            weight_np[mask_np == 0] = 1.0 + penalty[mask_np == 0]
        else:
            # 如果图里只有1个细胞或没细胞，不存在相互粘连的问题
            weight_np = np.ones_like(mask_np, dtype=np.float32)
        """

        # =====================================================================
        # 【当前运行方案】
        # 适用场景：语义分割 (VOC2012)，无法区分个体，改为对所有物体边缘施加惩罚
        # =====================================================================

        # 提取二值掩码：前景(物体)设为 1，背景(0)和白边(255)设为 0
        binary_mask = np.zeros_like(mask_np, dtype=np.float32)
        binary_mask[(mask_np > 0) & (mask_np < 255)] = 1.0

        if binary_mask.sum() == 0 or binary_mask.sum() == binary_mask.size:
            weight_tensor = torch.ones_like(mask_tensor, dtype=torch.float32)
        else:
            dist_to_bg = distance_transform_edt(binary_mask)
            dist_to_fg = distance_transform_edt(1.0 - binary_mask)

            # 获取每个像素到“交界边缘”的距离
            distance_to_border = dist_to_bg + dist_to_fg

            w0 = 10.0
            sigma = 5.0

            # 指数衰减惩罚
            weight_np = 1.0 + w0 * np.exp(- (distance_to_border ** 2) / (2 * sigma ** 2))

            # VOC2012特有：将 255 (忽略白边) 的权重强制设为 0，防止干扰
            weight_np[mask_np == 255] = 0.0

            weight_tensor = torch.from_numpy(weight_np).float()

        return img_tensor, mask_tensor, weight_tensor