import argparse
import torch

# 导入刚才写的两个工作模块
from train import train_model
from test import test_model

def get_model(model_name, device):
    """根据参数动态实例化模型"""
    if model_name == 'fcn32s':
        from models.fcn32s import FCN32s
        return FCN32s(num_classes=21).to(device)
    elif model_name == 'fcn16s':
        from models.fcn16s import FCN16s
        return FCN16s(num_classes=21).to(device)
    elif model_name == 'fcn8s':
        from models.fcn8s import FCN8s
        return FCN8s(num_classes=21).to(device)
    else:
        raise ValueError("不支持的模型！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FCN 训练与测试平台")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='选择运行模式: train 或 test')
    parser.add_argument('--model', type=str, default='fcn8s', choices=['fcn32s', 'fcn16s', 'fcn8s'], help='选择模型')
    parser.add_argument('--epochs', type=int, default=5, help='训练的总 Epoch 数')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch Size')

    parser.add_argument('--weight', type=str, default='',
                        help='测试时使用的权重文件路径 (例如: checkpoints/fcn32s_best.pth)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"👉 调度中心启动 | 模式: {args.mode.upper()} | 模型: {args.model} | 设备: {device}")

    # 实例化模型
    model = get_model(args.model, device)

    # 根据 mode 参数，将任务分发给对应的模块！
    if args.mode == 'train':
        train_model(args, model, device)
    elif args.mode == 'test':
        test_model(args, model, device)