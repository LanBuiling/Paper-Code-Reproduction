import torch
import torch.nn as nn
from torchvision import models


class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()

        vgg16 = models.vgg16(weights='DEFAULT')
        features = list(vgg16.features.children())

        # 截取到 pool3 (缩小 8 倍)
        self.pool3 = nn.Sequential(*features[:17])
        # 截取 pool3 到 pool4 (缩小 16 倍)
        self.pool4 = nn.Sequential(*features[17:24])
        # 截取 pool4 到 pool5 (缩小 32 倍)
        self.pool5 = nn.Sequential(*features[24:])

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # 为三层分别打分
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        # 上采样层
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1,
                                                bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        p3 = self.pool3(x)
        p4 = self.pool4(p3)
        p5 = self.pool5(p4)

        fc7 = self.drop7(self.relu7(self.fc7(self.drop6(self.relu6(self.fc6(p5))))))
        score_fr = self.score_fr(fc7)

        # 第一次融合: fc7 放大两倍 + pool4
        upscore2 = self.upscore2(score_fr)
        score_p4 = self.score_pool4(p4)
        fuse_pool4 = upscore2 + score_p4

        # 第二次融合: 刚刚融合的特征再放大两倍 + pool3
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_p3 = self.score_pool3(p3)
        fuse_pool3 = upscore_pool4 + score_p3

        # 最终放大 8 倍还原
        out = self.upscore8(fuse_pool3)
        return out