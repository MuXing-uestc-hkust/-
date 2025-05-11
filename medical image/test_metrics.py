import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import directed_hausdorff
import cv2
#import matplotlib
#matplotlib.use('Agg')  # 强制使用非交互式后端
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet
from fcn import FCN
from swin_unet import SwinUnet
from dataset import SegmentationDataset
import matplotlib.pyplot as plt

# 加载 VOC2012 数据集
def load_data(data_dir):
    """
    加载VOC2012数据集中的图像和标注。
    data_dir：VOC2012数据集路径
    split：选择的数据集分割，'test'
    """
    # 根据split加载图像和标注
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'labels')

    image_paths = []
    mask_paths = []

    # 获取测试集文件路径
    image_ids = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.png')]

    for img_id in image_ids:
        image_paths.append(os.path.join(images_dir, f'{img_id}.png'))
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
def compute_metrics(model, image_paths, mask_paths, num_classes=2):
    dice_scores = np.zeros(num_classes)
    hd_scores = np.zeros(num_classes)
    sample_counts = np.zeros(num_classes)  # 记录每个类别出现的次数
    num_samples = len(image_paths)

    for img_path, mask_path in zip(image_paths, mask_paths):
        # 读取图像和掩膜
        #img=np.load(img_path)
        img=cv2.imread(img_path)
        mask=Image.open(mask_path)
        mask=np.array(mask)/255
        #mask=np.load(mask_path)
        img= cv2.resize(img, (448, 448))
        #img = Image.open(img_path)
        #mask = Image.open(mask_path)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1).to(device)  # 变为 (3, 512, 512)

        # 调整大小时，使用最近邻插值方法
        #mask = mask.resize((384, 384), Image.NEAREST)
        mask = cv2.resize(mask, (448, 448), interpolation=cv2.INTER_NEAREST)

        # 将PIL图像转换为NumPy数组
        #mask = np.array(mask)
        #img = img.resize((512, 512))

        # 图像预处理：调整大小、归一化等（根据模型要求）
        img_tensor = preprocess_image(img).to(device)  # 例如：resize, normalize

        # 模型推理
        with torch.no_grad():
            model.eval()
            output = model(img_tensor.unsqueeze(0))  # 假设模型输出为分类后的像素概率
            print(output.shape)
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
    data_dir = '/hpc2hdd/home/xingmu/MedSAM/data/test_brain'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model = UNet(in_channels=3, out_channels=2).to(device)
    model = FCN(num_classes=2).to(device)

    #model = SwinUnet(img_size=448,num_classes=2,).to(device)
    # 加载测试集
    image_paths, mask_paths = load_data(data_dir)

    # 加载模型权重
    #state_dict = torch.load('checkpoints/unet_tumor_last.pth', map_location="cuda")
    #state_dict = torch.load('checkpoints/SwinUNet_tumor_last.pth', map_location="cuda")
    state_dict = torch.load('checkpoints/fcn_tumor_last.pth', map_location="cuda")
    #state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    # 计算所有类别的 Dice 和 Hausdorff 距离
    per_class_dice, per_class_hd, avg_dice, avg_hd = compute_metrics(model, image_paths, mask_paths)

    print("Per-class Dice Coefficients:", per_class_dice)
    print("Per-class Hausdorff Distances:", per_class_hd)
    print("Average Dice:", avg_dice)
    print("Average Hausdorff Distance:", avg_hd)

