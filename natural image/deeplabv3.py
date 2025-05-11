import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeeplabV3(nn.Module):
    def __init__(self, num_classes=21):
        """
        初始化 DeeplabV3 模型
        :param num_classes: 输出类别数，默认 21 类（VOC 数据集）
        """
        super(DeeplabV3, self).__init__()
        
        # 使用预训练的 ResNet101 作为主干网络
        self.model = segmentation.deeplabv3_resnet101(pretrained=True)
        
        # 修改分类器，以适应不同类别数
        self.model.classifier = DeepLabHead(2048, num_classes)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入图像
        :return: 输出预测图（分割图）
        """
        return self.model(x)
