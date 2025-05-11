import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 加载 .npy 文件
import cv2
label_data = np.load("/hpc2hdd/home/xingmu/MedSAM/data/npy_test/CT_Abd/gts/CT_Abd_FLARE22_Tr_0048-070.npy")  # 假设标签图像也以 npy 格式保存

# 创建标签到颜色的映射 (这里用随机颜色映射，仅作示例)
# 你可以根据需要自定义颜色
label_colors = {
    0: (0, 0, 0),  # 背景
    1: (255, 0, 0),  # 标签1 - 红色
    2: (0, 255, 0),  # 标签2 - 绿色
    3: (0, 0, 255),  # 标签3 - 蓝色
    4: (255, 255, 0),  # 标签4 - 黄色
    5: (0, 255, 255),  # 标签5 - 青色
    6: (255, 0, 255),  # 标签6 - 品红色
    7: (192, 192, 192),  # 标签7 - 灰色
    8: (128, 128, 0),  # 标签8 - 橄榄色
    9: (128, 0, 128),  # 标签9 - 紫色
    10: (0, 128, 128),  # 标签10 - 深青色
    11: (255, 165, 0),  # 标签11 - 橙色
    12: (255, 105, 180),  # 标签12 - 热粉色
    13: (255, 215, 0),  # 标签13 - 金色
}

# 初始化一个空白的 RGB 图像
rgb_label_image = np.zeros((label_data.shape[0], label_data.shape[1], 3), dtype=np.uint8)

# 将标签映射到对应的颜色
for label, color in label_colors.items():
    rgb_label_image[label_data == label] = color

image_data = np.load('/hpc2hdd/home/xingmu/MedSAM/data/npy_test/CT_Abd/imgs/CT_Abd_FLARE22_Tr_0048-070.npy')
image_data = (image_data * 255).astype(np.uint8)
alpha = 0.5  # 设置透明度，0为完全透明，1为完全不透明
overlay = cv2.addWeighted(image_data, 1 - alpha, rgb_label_image, alpha, 0)
cv2.imwrite('show/label3.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
