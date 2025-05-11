import os
import re
import cv2
import numpy as np
from skimage import measure
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def compute_boundary_f_score(pred, gt, tolerance=7):
    pred_boundary = extract_boundary(pred)
    gt_boundary = extract_boundary(gt)

    pred_dist = distance_transform_edt(1 - pred_boundary)
    gt_dist = distance_transform_edt(1 - gt_boundary)

    pred_match = (gt_dist < tolerance) * pred_boundary
    gt_match = (pred_dist < tolerance) * gt_boundary

    precision = pred_match.sum() / (pred_boundary.sum() + 1e-7)
    recall = gt_match.sum() / (gt_boundary.sum() + 1e-7)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def extract_boundary(mask):
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
    boundary = np.zeros_like(mask, dtype=np.uint8)
    for contour in contours:
        contour = np.round(contour).astype(np.int32)
        for y, x in contour:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                boundary[y, x] = 1
    return boundary

def extract_number(filename):
    # 提取文件名中的数字，例如 output_image_23.png -> 23
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def evaluate_j_and_f(pred_dir, gt_dir):
    pred_files = sorted(os.listdir(pred_dir), key=extract_number)
    gt_files = sorted(os.listdir(gt_dir), key=extract_number)

    assert len(pred_files) == len(gt_files), "预测和标签图像数量不一致"

    j_scores = []
    f_scores = []

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(gt_files), desc="Evaluating"):
        pred = cv2.imread(os.path.join(pred_dir, pred_file), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)
        print(pred_file)
        print(gt_file)
        #pred = (pred > 127).astype(np.uint8)
        #gt = (gt > 127).astype(np.uint8)

        j = compute_iou(pred, gt)
        f = compute_boundary_f_score(pred, gt)

        j_scores.append(j)
        f_scores.append(f)

    mean_j = np.mean(j_scores)
    mean_f = np.mean(f_scores)

    print(f"Mean J (IoU): {mean_j:.4f}")
    print(f"Mean F (Boundary F-score): {mean_f:.4f}")
    print(f"J&F Mean: {(mean_j + mean_f)/2:.4f}")

    return mean_j, mean_f
pred_dir = '/hpc2hdd/home/xingmu/MedSAM/computer_vision/yolov10_detection_and_tracking/results/hike'
gt_dir = "/hpc2hdd/home/xingmu/MedSAM/DAVIS/Annotations/1080p/hike"
evaluate_j_and_f(pred_dir, gt_dir)
