import os
from PIL import Image

# 定义原始目录和目标格式
image_root = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images"  # 根目录
sub_dirs = ["train", "val"]  # 子目录

# 遍历 train 和 val 目录
for sub_dir in sub_dirs:
    image_dir = os.path.join(image_root, sub_dir)
    for file in os.listdir(image_dir):
        if file.endswith(".tif"):
            tif_path = os.path.join(image_dir, file)
            png_path = os.path.join(image_dir, os.path.splitext(file)[0] + ".png")

            # 读取 .tif 并转换为 .png
            with Image.open(tif_path) as img:
                img.save(png_path, format="PNG")

            # 删除原 .tif 文件（可选）
            os.remove(tif_path)
            print(f"转换: {tif_path} -> {png_path}")

print("所有 .tif 文件已转换为 .png！")
