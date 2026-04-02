# utils/losses.py
import torch
import torch.nn as nn


class UNetWeightedLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(UNetWeightedLoss, self).__init__()
        self.ignore_index = ignore_index
        # reduction='none' 依然必须保留，因为要保留矩阵形状来乘权重图
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, predictions, targets, weight_maps):
        # 1. 计算基础的交叉熵损失矩阵 [Batch, H, W]
        # 此时 targets 中为 255 的位置，unweighted_loss 对应位置已经是 0 了
        unweighted_loss = self.ce(predictions, targets)

        # 2. 逐像素乘以你的终极惩罚图 (包含 wc(x) 和 指数距离惩罚)
        weighted_loss = unweighted_loss * weight_maps

        # 3. 【填坑操作】：找到所有不是 255 的有效像素点
        valid_mask = (targets != self.ignore_index)

        # 4. 只把有效像素的 Loss 挑出来
        valid_loss = weighted_loss[valid_mask]

        # 5. 安全地求均值 (防止某张图恰好全是 255 导致除以 0 报错)
        if valid_loss.numel() > 0:
            return valid_loss.mean()
        else:
            # 如果没有有效像素，返回一个带梯度的 0
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)