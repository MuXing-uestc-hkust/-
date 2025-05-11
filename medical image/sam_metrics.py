import os
import cv2
import numpy as np
from segment_anything2 import sam_model_registry, SamPredictor
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# ------------- 配置参数 -------------
test_image_dir = "/hpc2hdd/home/xingmu/MedSAM/data/test_data/images"  # 测试集图片路径
test_label_dir = "/hpc2hdd/home/xingmu/MedSAM/data/npy_test/CT_Abd/gts"  # 真实标签路径（npy格式）
#sam_checkpoint = "/hpc2hdd/home/xingmu/MedSAM/work_dir/MedSAM-ViT-B-20250331-1419/medsam_model_latest.pth"
sam_checkpoint = '/hpc2hdd/home/xingmu/MedSAM/work_dir/MedSAM-ViT-B-20250418-2302/medsam_model_best.pth'
#sam_checkpoint='medsam_vit_b.pth'
#sam_checkpoint='sam_vit_b_01ec64.pth'
yolo_model_path = "/hpc2hdd/home/xingmu/MedSAM/runs/detect/train/weights/best.pt"  # YOLO权重
#/hpc2hdd/home/xingmu/MedSAM/runs/detect/train9/weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- 加载模型 -------------
print("Loading SAM model...")
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
print(sam)
predictor = SamPredictor(sam)

print("Loading YOLO model...")
yolo_model = YOLO(yolo_model_path)

# ------------- 统计变量 -------------
num_classes = 14  # 总共14个类别（包含背景0）
dice_scores = {i: [] for i in range(1, num_classes)}  # 只存放 1~13 类
hd_scores = {i: [] for i in range(1, num_classes)}  # 只存放 1~13 类

# ------------- 处理测试集 -------------
test_images = sorted(os.listdir(test_image_dir))

for img_name in test_images:
    if not img_name.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片
        continue

    img_path = os.path.join(test_image_dir, img_name)
    label_path = os.path.join(test_label_dir, img_name.replace('.png', '.npy'))  # 假设标签是 .npy 文件

    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found, skipping...")
        continue

    print(f"Processing: {img_name}")

    # 读取图像
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 读取真实标签
    label_image = np.load(label_path)  # 假设 shape 为 (H, W)，值域为 0-13

    # 预测目标检测框
    results = yolo_model(img_path)
    result = results[0]
    boxes = result.boxes

    # 设置SAM图像
    predictor.set_image(image)

    # 初始化预测标签图像
    pred_label_image = np.zeros_like(label_image, dtype=np.uint8)

    # 遍历目标框进行分割
    for box in boxes:
        box_coords = box.xyxy.cpu().numpy()
        class_id = int(box.cls.cpu().numpy()) + 1  # +1 使其匹配 label 映射

        # SAM 预测
        masks, _, _ = predictor.predict(
            point_coords=None, point_labels=None, box=box_coords, multimask_output=False
        )
        mask = masks[0]  # 选择第一张 mask 作为预测结果

        # 更新预测标签
        pred_label_image[mask] = class_id

    # ------------- 计算 Dice 系数 -------------
    for class_id in range(1, num_classes):  # 跳过背景类 0
        pred_mask = (pred_label_image == class_id).astype(np.uint8)
        gt_mask = (label_image == class_id).astype(np.uint8)

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)

        if union > 0:
            dice = 2 * intersection / union
            dice_scores[class_id].append(dice)
            print(dice)

    # ------------- 计算 Hausdorff 距离 (HD) -------------
    for class_id in range(1, num_classes):  # 跳过背景类 0
        pred_points = np.column_stack(np.where(pred_label_image == class_id))
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

