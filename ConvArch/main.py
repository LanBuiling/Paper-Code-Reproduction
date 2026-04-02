# main.py
import argparse
import sys
import os

# 确保工程根目录在路径中，防止导入失败
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="ConvArch 卷积架构复现平台")

    # 1. 核心选择参数
    parser.add_argument('--model', type=str, required=True,
                        choices=['unet', 'unet_original', 'unet_vgg', 'fcn'],
                        help='选择要运行的模型')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式：训练或测试')

    # 2. 通用训练超参数
    parser.add_argument('--data_path', type=str, default='./dataset/VOCdevkit/VOC2012', help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', help='cuda 或 cpu')
    parser.add_argument('--early', type=int, default=3, help='早停轮数')

    args = parser.parse_args()

    # 3. 分发逻辑
    if args.model in ['unet', 'unet_original']:
        if args.mode == 'train':
            # 动态导入，避免一开始就加载所有模型占用内存
            from trains.UNet.unet_trainer import UNetTrainer
            trainer = UNetTrainer(args)
            trainer.train()
        elif args.mode == 'test':
            # 引入刚刚写好的 Tester
            from tests.UNet.unet_tester import UNetTester
            tester = UNetTester(args)
            tester.test()
        else:
            # from tests.UNet.unet_tester import UNetTester
            # tester = UNetTester(args)
            # tester.test()
            print("其他模块待开发...")

    elif args.model == 'fcn':
        print("FCN 架构待复现...")

    elif args.model == 'unet_vgg':
        if args.mode == 'train':
            from trains.UNet.unet_smp_trainer import SMPTrainer
            trainer = SMPTrainer(args)
            trainer.train()
        elif args.mode == 'test':
            # 新增这三行
            from tests.UNet.unet_smp_tester import SMPTester
            tester = SMPTester(args)
            tester.test()


if __name__ == '__main__':
    main()