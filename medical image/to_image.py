import os
import cv2
import shutil
import numpy as np
# 定义输入数据集目录和输出目录
data_dir = "/hpc2hdd/home/xingmu/MedSAM/data/brain_tumor"  # 原始数据集目录
output_train_image_dir = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/train"
output_train_mask_dir="/hpc2hdd/home/xingmu/MedSAM/data/dataset1/labels/train"
output_val_image_dir = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val"
output_val_mask_dir = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/labels/val"
# 创建输出目录

def process_images(input_dir, output_img_dir, output_mask_dir):
    """
    处理数据集：
    1. 将原图像转换为灰度图，并存入 output_img_dir
    2. 直接复制掩码图像到 output_mask_dir
    """
    for patient_folder in os.listdir(input_dir):
        patient_path = os.path.join(input_dir, patient_folder)

        if not os.path.isdir(patient_path):  # 确保是目录
            continue

        for file in os.listdir(patient_path):
            file_path = os.path.join(patient_path, file)
            file_lower = file.lower()

            # 先判断是否是掩码文件
            if 'mask' in file_lower and file_lower.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                # 复制掩码图像
                output_mask_path = os.path.join(output_mask_dir, file)
                shutil.copy(file_path, output_mask_path)
                continue  # 处理完掩码后直接跳过，不进行灰度转换

            # 处理普通原图像
            if file_lower.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
                image = np.repeat(image[:, :, None], 3, axis=-1)  # 扩展为 3 通道

                # 保存灰度图像
                output_img_path = os.path.join(output_img_dir, file)
                cv2.imwrite(output_img_path, image)


# 处理 train 和 val 目录
process_images(os.path.join(data_dir, "train"),output_train_image_dir,output_train_mask_dir)

process_images(os.path.join(data_dir, "val"), output_val_image_dir,output_val_mask_dir)

print("所有图像处理完成！")
