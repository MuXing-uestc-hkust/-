import torch
#import matplotlib
import cv2
#matplotlib.use('Agg')  # 强制使用非交互式后端
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from unet import UNet
from fcn import FCN
from swin_unet import SwinUnet
import numpy as np
def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # 去掉 'module.' 前缀
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict
# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = 14  # 假设VOC2012数据集有21个类别

# 加载训练好的U-Net模型
model = UNet(in_channels=3, out_channels=num_classes).to(device)
#model=DeeplabV3(num_classes=14)
#model = FCN(num_classes).to(device)
#model = SwinUnet(
       # img_size=448,
        #num_classes=14,  # 比如2类分割
   # ).to(device)
#model.load_state_dict(torch.load("unet_final.pth", map_location=torch.device('cpu')))
state_dict = torch.load('checkpoints/unet_tumor_last.pth', map_location="cpu")
#state_dict = torch.load('/Users/muxing/PycharmProjects/pythonProject2/IJCNN/unet_final.pth', map_location="cpu")
#state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
#model.load_state_dict(torch.load("unet_final.pth", map_location=torch.device('cpu')))
model.eval()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((448,448 )),  # 根据需要调整大小
    transforms.ToTensor(),  # 转为Tensor格式
])

# 加载并处理单张测试图像
image_path = '/hpc2hdd/home/xingmu/MedSAM/data/test_data/images/CT_Abd_FLARE22_Tr_0048-070.png'
#image_path ='/Users/muxing/PycharmProjects/pythonProject2/IJCNN/voc_2012_segmentation_data/train_val_images/2007_000032.jpg'# 修改为你的测试图像路径
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0).to(device)

# 执行推理
with torch.no_grad():
    output = model(input_image)
    preds = torch.argmax(output, dim=1).cpu().numpy()  # 获取预测类别（最大概率对应的类别）

# 定义 VOC 2012 的颜色映射（类别到 RGB 的映射）
label_colors = {
    0: (0, 0, 0),  # 背景
    1: (255, 0, 0),  # 红色
    2: (0, 255, 0),  # 绿色
    3: (0, 0, 255),  # 蓝色
    4: (255, 255, 0),  # 黄色
    5: (0, 255, 255),  # 青色
    6: (255, 0, 255),  # 品红色
    7: (192, 192, 192),  # 灰色
    8: (128, 128, 0),  # 橄榄色
    9: (128, 0, 128),  # 紫色
    10: (0, 128, 128),  # 深青色
    11: (255, 165, 0),  # 橙色
    12: (255, 105, 180),  # 热粉色
    13: (255, 215, 0),  # 金色
}

# 将预测的类别转换为 RGB 图像
#segmentation_image = Image.new("RGB", (preds.shape[2], preds.shape[1]))

# 创建一个列表用于存放 RGB 数据
segmentation_data = []

for label in preds[0].flatten():
    segmentation_data.append(label_colors[label])
segmentation_image=np.array(segmentation_data)
segmentation_image = segmentation_image.reshape((448, 448, 3))
segmentation_image = cv2.resize( segmentation_image, (512, 512),interpolation=cv2.INTER_NEAREST).astype(np.uint8)

# 将数组转换为 PIL 图像
segmentation_image = Image.fromarray(np.uint8(segmentation_image))
image=np.array(image)
segmentation_image=np.array(segmentation_image)
# 将 RGB 数据转换为图像
alpha = 0.5  # 设置透明度，0为完全透明，1为完全不透明
overlay = cv2.addWeighted(image, 1 - alpha, segmentation_image, alpha, 0)
# 可视化原图与分割结果
cv2.imwrite('show/swin_unet1.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
