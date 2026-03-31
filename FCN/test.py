import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.dataLoad.dataloader import VOCSegmentationDataset


# ---------------------------------------------------------
# PASCAL VOC 标准 21 类调色板
# ---------------------------------------------------------
def decode_segmap(pred_tensor):
    """将 [H, W] 的类别索引矩阵，映射成 [H, W, 3] 的 RGB 彩色图片"""
    colors = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
        [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
        [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
    ], dtype=np.uint8)

    r = np.zeros_like(pred_tensor, dtype=np.uint8)
    g = np.zeros_like(pred_tensor, dtype=np.uint8)
    b = np.zeros_like(pred_tensor, dtype=np.uint8)

    for l in range(0, 21):
        idx = pred_tensor == l
        r[idx] = colors[l, 0]
        g[idx] = colors[l, 1]
        b[idx] = colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb)


# ---------------------------------------------------------
# 核心测试逻辑
# ---------------------------------------------------------
def test_model(args, model, device):
    print(f"👀 开始执行半透明叠加测试: 模型={args.model}")

    if not args.weight or not os.path.exists(args.weight):
        raise ValueError(f"❌ 找不到权重文件！请使用 --weight 参数指定正确的路径。")

    model.load_state_dict(torch.load(args.weight, map_location=device))
    print(f"✅ 成功加载预训练权重: {args.weight}")

    test_dataset = VOCSegmentationDataset(root_dir='./dataset/VOCdevkit/VOC2012', image_set='val', crop_size=(320, 320))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            real_file_name = test_dataset.file_names[i]

            # 1. 模型推理
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            pred_mask = preds[0]

            # 2. 生成预测的彩色蒙版
            seg_img = decode_segmap(pred_mask)

            # ---------------------------------------------------------
            # 【新增：原图逆向还原与融合叠加】
            # ---------------------------------------------------------
            # 因为 dataloader 里用 TF.to_tensor 把像素除以了 255 变成了小数
            # 我们现在要把它乘回 255，并把形状从 [3, H, W] 换回 [H, W, 3]
            img_tensor = images[0].cpu()
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            orig_img = Image.fromarray(img_np)

            # 使用 PIL 的 blend 函数进行融合！alpha=0.6 表示蒙版占 60% 的不透明度
            blended_img = Image.blend(orig_img, seg_img, alpha=0.8)

            # 3. 保存融合后的结果
            res_path = os.path.join('results', f'{args.model}_{real_file_name}_overlay.png')
            blended_img.save(res_path)

            print(f"🎨 [{real_file_name}.jpg] 的半透明叠加图已生成: {res_path}")

            if i == 4:  # 测试 5 张退出
                break

    print("加载完毕！")