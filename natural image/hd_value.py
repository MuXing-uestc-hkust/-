from PIL import Image
import numpy as np
import numpy as np
from scipy.spatial.distance import cdist
segmentation_image=Image.open("/hpc2hdd/home/xingmu/IJCNN/ASAM/HD/Asam_火车.png")
segmentation_image=np.array(segmentation_image)
label_image=Image.open("/hpc2hdd/home/xingmu/IJCNN/ASAM/HD/2007_005689.png")
label_image=np.array(label_image)

def rgb_to_class_map(segmentation_map, class_colors):
    """
    将三通道的分割结果图 (RGB) 转换为类别表示 (单通道类别图)。

    :param segmentation_map: 分割结果图，形状为 [H, W, 3]
    :param class_colors: 每个类别对应的 RGB 值，形状为 [num_classes, 3]
    :return: 单通道类别图，形状为 [H, W]，每个像素值为类别编号
    """
    # 初始化类别图
    height, width, _ = segmentation_map.shape
    class_map = np.zeros((height, width), dtype=np.int32)

    # 遍历每个类别的颜色，将匹配的像素设置为该类别
    for class_idx, color in enumerate(class_colors):
        mask = np.all(segmentation_map == color, axis=-1)
        class_map[mask] = class_idx

    return class_map

# 类别的 RGB 对应表 (假设有 4 类)
class_colors = np.array([
   [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128], [224, 224, 192]

])
pre_class_map = rgb_to_class_map(segmentation_image, class_colors)
pred_cls = (pre_class_map == 19)
gt_cls = (label_image == 19)

    # 计算交集和并集
intersection = np.logical_and(pred_cls, gt_cls).sum()
union = np.logical_or(pred_cls, gt_cls).sum()

    # IoU
iou = intersection / union if union > 0 else 0

    # Dice
tp_fp_fn = pred_cls.sum() + gt_cls.sum()
dice = (2 * intersection) / tp_fp_fn if tp_fp_fn > 0 else 0

    # Hausdorff Distance (HD)
    # 只考虑非零部分，即边缘点
pred_indices = np.column_stack(np.where(pred_cls))
gt_indices = np.column_stack(np.where(gt_cls))

    # 如果没有边缘点，则 Hausdorff 距离为 0
if len(pred_indices) == 0 or len(gt_indices) == 0:
    hd = 0
else:
        # 计算每个预测点到真实标签的最短距离，以及反过来的距离
    dist_pred_to_gt = cdist(pred_indices, gt_indices, metric='euclidean')
    dist_gt_to_pred = cdist(gt_indices, pred_indices, metric='euclidean')

        # 计算 HD
    hd = max(dist_pred_to_gt.min(axis=1).max(), dist_gt_to_pred.min(axis=1).max())
print(hd)
print(dice)
