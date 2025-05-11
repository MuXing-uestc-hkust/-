import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet
from models.fcn import FCN
from deeplabv3 import DeeplabV3
from dataset import SegmentationDataset
import matplotlib.pyplot as plt
from swin_unet import SwinUnet
# 配置
batch_size = 32
epochs = 100
learning_rate = 1e-3
image_size = (384, 384)
num_classes = 21  # 假设有20个类别+1个背景类

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = SegmentationDataset('voc_2012_segmentation_data/train_val_images',
                                    'voc_2012_segmentation_data/train_val_labels', transform=transform)
#valid_dataset = SegmentationDataset('voc_2012_segmentation_data/valid_images',
                                  #  'voc_2012_segmentation_data/valid_labels', transform=transform)
test_dataset = SegmentationDataset('voc_2012_segmentation_data/test_images', 'voc_2012_segmentation_data/test_labels',
                                   transform=transform)  # 加载测试集

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型、损失函数、优化器
#model = UNet(in_channels=3, out_channels=num_classes).to(device)
#model = DeeplabV3(num_classes=21).to(device)
model=SwinUnet(embed_dim=96,patch_height=4,patch_width=4, class_num=21).to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
#model = FCN(num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()  # 多类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 存储损失
train_losses = []
valid_losses = []

# 初始化最小验证损失
best_valid_loss = float('inf')

# 训练循环
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels[:, 0, :, :]  # 结果形状为 (8, 256, 256)
        labels = labels.long()
        optimizer.zero_grad()
        labels[labels == 255] = 0
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算训练损失
        train_loss += loss.item()

    # 打印结果
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}')

    # 存储训练损失
    train_losses.append(train_loss / len(train_loader))

    # 验证集计算
    #model.eval()
    #valid_loss = 0.0
    #with torch.no_grad():
        #for images, labels in valid_loader:
          #  images, labels = images.to(device), labels.to(device)
            #labels = labels[:, 0, :, :]
            #labels = labels.long()
            #labels[labels == 255] = 0
            #outputs = model(images)
            #loss = criterion(outputs, labels)
            #valid_loss += loss.item()

    # 打印验证集损失
    #print(f'Valid Loss: {valid_loss / len(valid_loader):.4f}')

    # 存储验证损失
    #valid_losses.append(valid_loss / len(valid_loader))

    # 如果当前验证损失更小，则保存模型
    # if valid_loss < best_valid_loss:
    #     best_valid_loss = valid_loss
    #     best_epoch = epoch
    #     torch.save(model.state_dict(), "best_unet_model.pth")  # 保存验证集损失最小的模型
    #     print(f"Model saved at epoch {epoch + 1} with validation loss {valid_loss / len(valid_loader):.4f}")

# 保存最终模型
torch.save(model.state_dict(), "swinunet_final.pth")
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.savefig('training_loss_curve_swinet.png')  # 保存图像到文件
print("Training loss curve saved as 'swinunet_training_loss_curve.png'.")



