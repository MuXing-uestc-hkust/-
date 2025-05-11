import numpy as np
import os
from pathlib import Path


def convert_to_yolo_format(gt2D, image_shape):
    """
    将标签图像(gt2D)转换为 YOLO 格式
    :param gt2D: 目标标签图像，二维数组，每个像素值代表类别 ID
    :param image_shape: 图像的尺寸 (H, W)
    :return: 返回每个目标框的 YOLO 格式标签
    """
    image_height, image_width = image_shape
    yolo_labels = []

    # 获取每个类别的目标区域
    for class_id in np.unique(gt2D):  # 获取标签中的所有类别
        if class_id == 0:  # 背景类别跳过
            continue

        # 获取当前类别的像素位置
        y_indices, x_indices = np.where(gt2D == class_id)

        if len(x_indices) == 0:  # 没有目标区域
            continue

        # 计算目标区域的边界框
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # 计算 YOLO 格式的框坐标
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # 将坐标归一化到 [0, 1] 范围
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        # 保存 YOLO 格式的标签
        yolo_labels.append(f"{class_id - 1} {x_center} {y_center} {width} {height}")

    return yolo_labels


def save_yolo_labels(output_dir, img_name, yolo_labels):
    """
    将 YOLO 格式标签保存为文本文件
    :param output_dir: 保存标签文件的目录
    :param img_name: 图像文件名（不带扩展名）
    :param yolo_labels: 转换后的 YOLO 格式标签
    """
    os.makedirs(output_dir, exist_ok=True)
    label_file = os.path.join(output_dir, img_name + '.txt')
    with open(label_file, 'w') as f:
        for label in yolo_labels:
            f.write(label + '\n')


def convert_all_labels(input_dir, output_dir, image_shape):
    """
    批量转换目录下所有的.npy标签文件为YOLO格式
    :param input_dir: 输入的标签目录，包含.npy格式标签
    :param output_dir: 输出的标签目录，保存YOLO格式标签
    :param image_shape: 图像的尺寸 (H, W)
    """
    npy_files = Path(input_dir).rglob("*.npy")  # 遍历标签文件夹下所有.npy文件

    for npy_file in npy_files:
        gt2D = np.load(npy_file)  # 加载标签数据
        img_name = npy_file.stem  # 获取文件名（不带扩展名）

        # 将标签转换为YOLO格式
        yolo_labels = convert_to_yolo_format(gt2D, image_shape)

        # 保存YOLO格式标签
        save_yolo_labels(output_dir, img_name, yolo_labels)
        print(f"Converted {img_name} to YOLO format.")


# 示例用法
input_dir = '/hpc2hdd/home/xingmu/MedSAM/data/npy/CT_Abd/gts'  # 包含 .npy 标签的文件夹
output_dir = '/hpc2hdd/home/xingmu/MedSAM/data/yolo_labels'  # 输出 YOLO 标签的文件夹
image_shape = (512,512)  # 图像的尺寸 (H, W)

convert_all_labels(input_dir, output_dir, image_shape)
