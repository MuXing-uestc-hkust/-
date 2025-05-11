import cv2
import numpy as np
from segment_anything3 import sam_model_registry, SamPredictor
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from PIL import Image
# 初始化SAM模型
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
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
    #image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

image = cv2.imread('/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/images/cju0qoxqj9q6s0835b43399p4.jpg')
image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_path="/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/images/cju0qoxqj9q6s0835b43399p4.jpg"
#/hpc2hdd/home/xingmu/MedSAM/data/test_data/images/CT_Abd_FLARE22_Tr_0048-070.png
#image_path = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val/TCGA_CS_4941_19960909_13.png"
#image_path = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val/TCGA_HT_A61B_19991127_35.png"
#image_path = "/hpc2hdd/home/xingmu/MedSAM/data/dataset1/images/val/TCGA_HT_A61B_19991127_46.png"
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sam_checkpoint="/hpc2hdd/home/xingmu/MedSAM/work_dir/MedSAM-ViT-B-20250426-2324/medsam_model_latest.pth"
#sam_checkpoint = "sam_vit_b_01ec64.pth"
#sam_checkpoint = 'medsam_vit_b.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)
predictor.set_image(image)

rgb_label_image = np.zeros((256, 256, 3), dtype=np.uint8)
model = YOLO("/hpc2hdd/home/xingmu/MedSAM/runs/detect/train10/weights/best.pt")  # load a custom model
results = model("/hpc2hdd/home/xingmu/MedSAM/data/polyp_test/images/cju0qoxqj9q6s0835b43399p4.jpg")
result=results[0]
boxes=result.boxes
for box in result.boxes:
    box_cood=box.xyxy.cpu().numpy()
    class_id=int(box.cls.cpu().numpy())+1
    conf=box.conf.cpu().numpy()[0]
    if conf<0.5:
        continue
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_cood,
        multimask_output=False,
    )
    mask=masks[0]
    #rgb_label_image[mask]=label_colors[class_id]
# 处理预测掩码
#pred_mask = preds[0]  # 取 batch 中的第一张图像
#pred_mask = (pred_mask > 0).astype(np.uint8) * 255  # 转换为二值掩码 (0/255)
mask=mask.astype(np.uint8)*255
# 结果保存路径
save_path_contours = "show/polyp_our_contours.png"  # 带红色边界的
save_path_overlay = "show/polyp_our_overlay.png"  # 纯色分割叠加
# 保存可视化分割结果
save_segmentation_results(image_path, mask, save_path_contours, save_path_overlay)




