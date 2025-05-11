import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import directed_hausdorff
from models.unet import UNet
from models.fcn import FCN
from deeplabv3 import DeeplabV3
from swin_unet import SwinUnet
import cv2


# 加载 VOC2012 数据集
def load_voc2012_data(data_dir, split='test'):
    """
    加载VOC2012数据集中的图像和标注。
    data_dir：VOC2012数据集路径
    split：选择的数据集分割，'test'
    """
    # 根据split加载图像和标注
    images_dir = os.path.join(data_dir, 'train_val_images')
    annotations_dir = os.path.join(data_dir, 'train_val_labels')

    image_paths = []
    mask_paths = []

    # 获取测试集文件路径
    image_ids = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.jpg')]

    for img_id in image_ids:
        image_paths.append(os.path.join(images_dir, f'{img_id}.jpg'))
        mask_paths.append(os.path.join(annotations_dir, f'{img_id}.png'))

    return image_paths, mask_paths


# 计算 Dice 系数
def dice_coefficient(pred_mask, true_mask, class_id):
    pred_bin = (pred_mask == class_id).astype(np.float32)
    true_bin = (true_mask == class_id).astype(np.float32)
    intersection = np.sum(pred_bin * true_bin)

    smooth = 1e-6  # 平滑因子
    return 2. * intersection / (np.sum(pred_bin) + np.sum(true_bin) + smooth)


# 计算 Hausdorff 距离
def hausdorff_distance(pred_mask, true_mask, class_id):
    pred_bin = (pred_mask == class_id)
    true_bin = (true_mask == class_id)

    # 提取边界坐标
    pred_coords = np.array(np.nonzero(pred_bin)).T
    true_coords = np.array(np.nonzero(true_bin)).T

    if len(pred_coords) == 0 or len(true_coords) == 0:
        return 0.0  # 无边界时 Hausdorff 距离为0

    # 计算 Hausdorff 距离
    forward_hd = directed_hausdorff(pred_coords, true_coords)[0]
    backward_hd = directed_hausdorff(true_coords, pred_coords)[0]

    return max(forward_hd, backward_hd)


# 计算所有类别的 Dice 和 HD
def compute_metrics(model, image_paths, mask_paths, num_classes=21):
    dice_scores = np.zeros(num_classes)
    hd_scores = np.zeros(num_classes)
    sample_counts = np.zeros(num_classes)  # 记录每个类别出现的次数
    num_samples = len(image_paths)

    for img_path, mask_path in zip(image_paths, mask_paths):
        # 读取图像和掩膜
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # 调整大小时，使用最近邻插值方法
        mask = mask.resize((384, 384), Image.NEAREST)

        # 将PIL图像转换为NumPy数组
        mask = np.array(mask)
        img = img.resize((384, 384))

        # 图像预处理：调整大小、归一化等（根据模型要求）
        img_tensor = preprocess_image(img).to(device)  # 例如：resize, normalize

        # 模型推理
        with torch.no_grad():
            model.eval()
            output = model(img_tensor.unsqueeze(0))  # 假设模型输出为分类后的像素概率
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]  # 预测掩膜

        # 对每个类别计算 Dice 和 HD
        for class_id in range(num_classes):
            # 计算 Dice 系数和 Hausdorff 距离
            dice_scores[class_id] += dice_coefficient(pred_mask, mask, class_id)
            hd_scores[class_id] += hausdorff_distance(pred_mask, mask, class_id)
            if np.any(mask == class_id):  # 如果真实标签中有该类别
                sample_counts[class_id] += 1

    # 计算平均 Dice 和 HD
    avg_dice = np.mean(dice_scores[1:] / sample_counts[1:])  # 排除背景类（class_id=0）
    avg_hd = np.mean(hd_scores[1:] / sample_counts[1:])

    # 计算每个类别的平均 Dice 和 HD
    per_class_dice = dice_scores / sample_counts
    per_class_hd = hd_scores / sample_counts

    return per_class_dice, per_class_hd, avg_dice, avg_hd


# 预处理图像：根据模型要求进行转换
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)


# 移除模型权重中的 'module.' 前缀
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉 'module.' 前缀
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


# 运行主流程
if __name__ == "__main__":
    # 假设VOC数据集存放路径
    data_dir = 'voc_2012_segmentation_data'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 使用不同的模型
    model = SwinUnet(embed_dim=96, patch_height=4, patch_width=4, class_num=21).to(device)
   # model = DeeplabV3(num_classes=21).to(device)

    # 加载测试集
    image_paths, mask_paths = load_voc2012_data(data_dir, split='test')

    # 加载模型权重
    state_dict = torch.load('swinunet_final.pth', map_location="cpu")
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    # 计算所有类别的 Dice 和 Hausdorff 距离
    per_class_dice, per_class_hd, avg_dice, avg_hd = compute_metrics(model, image_paths, mask_paths)

    print("Per-class Dice Coefficients:", per_class_dice)
    print("Per-class Hausdorff Distances:", per_class_hd)
    print("Average Dice:", avg_dice)
    print("Average Hausdorff Distance:", avg_hd)

