import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image

class MyFRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 images_dir, 
                 labels_dir, 
                 transforms=None, 
                 img_ext=".png"):
        """
        :param images_dir: 存放图像的文件夹路径
        :param labels_dir: 存放txt标签文件的文件夹路径
        :param transforms: 数据增强/预处理操作
        :param img_ext:   图像的扩展名
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.img_ext = img_ext
        
        # 获取所有图像文件的文件名(不含后缀)
        self.image_ids = [os.path.splitext(f)[0] 
                          for f in os.listdir(self.images_dir) 
                          if f.endswith(self.img_ext)]
        self.image_ids.sort()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1) 读取图像
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id + self.img_ext)
        label_path = os.path.join(self.labels_dir, image_id + ".txt")
        
        image = Image.open(img_path).convert("RGB")
        width, height = image.size  # 获取图像的宽高(绝对像素)
        
        # 2) 读取对应txt标签
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 标签格式: class_id x_center y_center w h (均是归一化)
                    class_id_str, xc_str, yc_str, w_str, h_str = line.split()
                    class_id = int(class_id_str)
                    x_center = float(xc_str)
                    y_center = float(yc_str)
                    w = float(w_str)
                    h = float(h_str)

                    # 转换为绝对坐标
                    xc_abs = x_center * width
                    yc_abs = y_center * height
                    w_abs = w * width
                    h_abs = h * height

                    # 计算xmin, ymin, xmax, ymax
                    xmin = xc_abs - w_abs / 2
                    ymin = yc_abs - h_abs / 2
                    xmax = xc_abs + w_abs / 2
                    ymax = yc_abs + h_abs / 2

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 3) 构建target字典 (Faster R-CNN需要：boxes, labels, image_id可选)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # 4) 若有自定义 transforms(如随机裁剪、翻转等)，可在此处理
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target
transform = transforms.Compose([
    transforms.ToTensor()
])
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes):
    # 1) 加载预训练的Faster R-CNN (ResNet50+FPN)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # 2) 替换分类器
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
import torch
from torch.utils.data import DataLoader

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        # 移动到GPU/CPU
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向 + 计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)


def main_train_loop(images_dir, labels_dir, num_classes, 
                    num_epochs=100, batch_size=12, lr=1e-3, device="cuda"):
    # 1) 准备数据集和DataLoader
    dataset = MyFRCNNDataset(
        images_dir=images_dir, 
        labels_dir=labels_dir,
        transforms=transforms.Compose([transforms.ToTensor()])
    )
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))  # 适配目标检测
    )

    # 2) 获取模型
    model = get_faster_rcnn_model(num_classes=num_classes)
    model.to(device)

    # 3) 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    
    # 4) 训练若干 epoch
    for epoch in range(num_epochs):
        loss_train = train_one_epoch(model, optimizer, data_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss_train:.4f}")

    # 5) 训练结束后，保存模型
    torch.save(model.state_dict(), "faster_rcnn_model.pth")
    print("训练完成并保存模型至 faster_rcnn_model.pth")

# 直接调用主训练函数示例
if __name__ == "__main__":
    images_dir = "/hpc2hdd/home/xingmu/MedSAM/data/dataset/images/train"
    labels_dir = "/hpc2hdd/home/xingmu/MedSAM/data/dataset/labels/train"
    num_classes = 14  # 假设你有12个目标类别(1~12)，再加背景 => 13
    main_train_loop(images_dir, labels_dir, num_classes)

