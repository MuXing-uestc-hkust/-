import numpy as np
import os
from pathlib import Path
import cv2


def convert_npy_to_png(input_dir, output_dir):
    """
    将.npy格式的图像转换为.png格式并保存
    :param input_dir: 输入目录，包含.npy文件
    :param output_dir: 输出目录，保存转换后的.png文件
    """
    # 获取所有.npy文件
    npy_files = Path(input_dir).rglob("*.npy")  # 遍历目录下所有.npy文件

    for npy_file in npy_files:
        # 读取.npy文件
        image_data = np.load(npy_file)

        # 将数据转换为0-255的范围
        image_data = np.uint8(image_data * 255)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 生成对应的.png文件路径
        img_name = npy_file.stem + '.png'
        output_path = os.path.join(output_dir, img_name)

        # 保存为PNG格式
        cv2.imwrite(output_path, image_data)

        print(f"Converted {npy_file} to {output_path}")


# 示例用法
input_dir = '/hpc2hdd/home/xingmu/MedSAM/data/npy_test/CT_Abd/imgs'  # 包含 .npy 图像的目录
output_dir = '/hpc2hdd/home/xingmu/MedSAM/data/test_data/images'  # 输出 .png 图像的目录

convert_npy_to_png(input_dir, output_dir)
