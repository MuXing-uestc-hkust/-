import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from unet import UNet
from fcn import FCN
from swin_unet import SwinUnet

def save_segmentation_results(test_image_path, pred_mask, save_path_contours, save_path_overlay):
    """
    读取原始图像，并分别保存：
    1. 叠加红色分割边界的原始图像
    2. 叠加纯白色分割区域的原始图像

    参数：
    - test_image_path: str, 原始测试图像路径
    - pred_mask: np.array, 预测的二分类掩码 (H, W)，值应为 0 或 255
    - save_path_contours: str, 结果保存路径（红色边界）
    - save_path_overlay: str, 结果保存路径（纯色分割）
    """
    # 读取原始图像
    image = cv2.imread(test_image_path)  # BGR 格式
    original_size = (image.shape[1], image.shape[0])  # (W, H)

    # 调整预测掩码大小，使其匹配原图尺寸
    pred_mask_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)

    # 确保 pred_mask_resized 是单通道二值图像 (0/255)
    pred_mask_resized = (pred_mask_resized > 0).astype(np.uint8) * 255

    # -------- 1. 生成红色边界叠加图像 --------
    contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    cv2.drawContours(image_contours, contours, -1, (0, 0, 255), 2)  # 绘制红色边界 (14, 7, 176), (28, 214, 28)
    cv2.imwrite(save_path_contours, image_contours)
    print(f"红色边界图像保存成功: {save_path_contours}")

    # -------- 2. 生成纯色分割叠加图像 --------
    overlay = image.copy()
    mask_color = np.zeros_like(image, dtype=np.uint8)
    mask_color[:, :, :] = (255, 255, 255)  # 纯白色区域

    # 只在 mask 区域内进行掩码叠加
    overlay[pred_mask_resized == 255] = mask_color[pred_mask_resized == 255]

    # 叠加透明度
    alpha = 1  # 透明度 (0-1)
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    cv2.imwrite(save_path_overlay, blended)
    print(f"纯色分割结果保存成功: {save_path_overlay}")

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes=2
# 加载模型
#model = UNet(in_channels=3, out_channels=2).to(device)
model = FCN(num_classes=2).to(device)
#model = SwinUnet(img_size=448,num_classes=2,).to(device)
#model.load_state_dict(torch.load("checkpoints/SwinUNet_tumor_last.pth", map_location=device))
model.load_state_dict(torch.load("checkpoints/fcn_polyp_last.pth", map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整大小
    transforms.ToTensor(),  # 转为Tensor格式
])

# 加载测试图像
#image_path = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val/TCGA_CS_4941_19960909_13.png"
#image_path = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val/TCGA_HT_A61B_19991127_35.png"
#image_path = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val/TCGA_HT_A61B_19991127_46.png"
image_path="/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/images/cju0qoxqj9q6s0835b43399p4.jpg"
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度
label_path="/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/masks/cju0qoxqj9q6s0835b43399p4.jpg"
# 执行推理
with torch.no_grad():
    output = model(input_image)  # 形状: (B, num_classes, H, W)
    preds = torch.argmax(output, dim=1).cpu().numpy()  # 变成 (B, H, W) 的 numpy 数组

# 处理预测掩码
pred_mask = preds[0]  # 取 batch 中的第一张图像
pred_mask = (pred_mask > 0).astype(np.uint8) * 255  # 转换为二值掩码 (0/255)

# 结果保存路径
save_path_contours = "show/polyp_fcn3_contours.png"  # 带红色边界的
save_path_overlay = "show/polyp_fcn3_overlay.png"  # 纯色分割叠加
label=cv2.imread(label_path)
label=Image.open(label_path)
label=np.array(label)
# 保存可视化分割结果
save_segmentation_results(image_path, pred_mask, save_path_contours, save_path_overlay)
