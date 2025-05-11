import os
import cv2
import numpy as np
from segment_anything1 import sam_model_registry, SamPredictor
from ultralytics import YOLO
import torch
from PIL import Image
import matplotlib.pyplot as plt
from unet import UNet
from fcn import FCN
from swin_unet import SwinUnet
from torchvision import transforms
from scipy.spatial.distance import directed_hausdorff

# ------------- 配置参数 -------------
test_image_dir = "/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/images"  # 测试集图片路径
test_label_dir = "/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/masks"  # 真实标签路径（npy格式）
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------- 统计变量 -------------
num_classes = 2  # 总共14个类别（包含背景0）
dice_scores = {i: [] for i in range(1, num_classes)}  # 只存放 1~13 类
hd_scores = {i: [] for i in range(1, num_classes)}  # 只存放 1~13 类

# ------------- 处理测试集 -------------
test_images = sorted(os.listdir(test_image_dir))
transform = transforms.Compose([
    transforms.Resize((512,512 )),  # 根据需要调整大小
    transforms.ToTensor(),  # 转为Tensor格式
])
for img_name in test_images:
    if not img_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片
        continue

    img_path = os.path.join(test_image_dir, img_name)
    label_path = os.path.join(test_label_dir, img_name)  # 假设标签是 .npy 文件

    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found, skipping...")
        continue

    print(f"Processing: {img_name}")

    # 读取图像
    image = Image.open(img_path)
    image = transform(image).unsqueeze(0).to(device)

    # 读取真实标签
    label_image = Image.open(label_path)# 假设 shape 为 (H, W)，值域为 0-13
    label_image=np.array(label_image)
    label_image[label_image<128]=0
    label_image[label_image>128]=1
    
    label_image = cv2.resize(label_image, (512, 512),interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    # 假设VOC数据集存放路径
    #model = UNet(in_channels=3, out_channels=2).to(device)
    model = FCN(num_classes=2).to(device)

    #model = SwinUnet(img_size=448,num_classes=2,).to(device)
    # 加载测试集

    # 加载模型权重
    #state_dict = torch.load('checkpoints/unet_polyp_last.pth', map_location="cuda")
    #state_dict = torch.load('checkpoints/swin_unet_polyp_last.pth', map_location="cuda")
    state_dict = torch.load('checkpoints/fcn_polyp_last.pth', map_location="cuda")
    #state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        output = model(image)  # 假设模型输出为分类后的像素概率
        pred_mask = torch.argmax(output, dim=1).cpu().numpy()[0]  # 预测掩膜
    
    # ------------- 计算 Dice 系数 -------------
    for class_id in range(1, num_classes):  # 跳过背景类 0
        #pred_mask = (pred_label_image == class_id).astype(np.uint8)
        gt_mask = (label_image == class_id).astype(np.uint8)

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)

        if union > 0:
            dice = 2 * intersection / union
            dice_scores[class_id].append(dice)
            print(dice)

    # ------------- 计算 Hausdorff 距离 (HD) -------------
    for class_id in range(1, num_classes):  # 跳过背景类 0
        pred_points = np.column_stack(np.where(pred_mask== class_id))
        gt_points = np.column_stack(np.where(label_image == class_id))

        if pred_points.shape[0] > 0 and gt_points.shape[0] > 0:
            hd = max(directed_hausdorff(pred_points, gt_points)[0], directed_hausdorff(gt_points, pred_points)[0])
            hd_scores[class_id].append(hd)

# ------------- 计算最终指标 -------------
final_dice = {i: np.mean(dice_scores[i]) if len(dice_scores[i]) > 0 else 0 for i in range(1, num_classes)}
final_hd = {i: np.mean(hd_scores[i]) if len(hd_scores[i]) > 0 else float('inf') for i in range(1, num_classes)}

# ------------- 输出结果 -------------
print("\n========== 最终评估结果（不包含背景类） ==========")
for class_id in range(1, num_classes):  # 跳过背景类 0
    print(f"Class {class_id}: Dice = {final_dice[class_id]:.4f}, HD = {final_hd[class_id]:.4f}")

average_dice = np.mean([v for v in final_dice.values() if v > 0])
average_hd = np.mean([v for v in final_hd.values() if v < float('inf')])

print(f"\nOverall Average Dice (excluding background): {average_dice:.4f}")
print(f"Overall Average HD (excluding background): {average_hd:.4f}")

