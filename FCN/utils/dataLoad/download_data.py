import os
from torchvision.datasets import VOCSegmentation


def download_voc():
    # 设置数据集文件夹
    root_dir = '../../dataset'

    # 确保 dataset 文件夹存在
    os.makedirs(root_dir, exist_ok=True)

    print("开始下载 PASCAL VOC 2012 数据集...")
    print("文件比较大（约 2GB），如果网速慢可能需要稍等片刻，请耐心等待~")

    # 使用 PyTorch 官方库自动下载并解压训练集
    train_dataset = VOCSegmentation(root=root_dir, year='2012', image_set='train', download=True)

    # 使用 PyTorch 官方库自动下载并解压验证集
    val_dataset = VOCSegmentation(root=root_dir, year='2012', image_set='val', download=True)

    print("\n🎉 PASCAL VOC 2012 数据集下载并解压完成！")


if __name__ == "__main__":
    download_voc()