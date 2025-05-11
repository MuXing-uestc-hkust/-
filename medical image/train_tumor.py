from dataset1 import braindataset
import numpy as np
from utils.metrics import dice_coeff
from augmentation import get_train_transforms, get_val_transforms
import torch
from unet import UNet
from fcn import FCN
import torch.nn as nn
import torch.optim as optim
from swin_unet import SwinUnet
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
device='cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_set=braindataset(root_dir='/hpc2hdd/home/xingmu/MedSAM/data/brain_tumor/train',transform=transform)
val_set=braindataset(root_dir='/hpc2hdd/home/xingmu/MedSAM/data/brain_tumor/val',transform=transform)
train_loader=DataLoader(train_set,batch_size=16,shuffle=True)
val_loader=DataLoader(val_set,batch_size=16,shuffle=False)
#model=UNet().to(device)
#model=FCN(num_classes=2).to(device)
#model = SwinUnet(
        #img_size=448,
        #num_classes=2,  # 比如2类分割
   # ).to(device)
#criterion=nn.BCELoss()
model = UNet(in_channels=3, out_channels=2).to(device)
criterion = torch.nn.CrossEntropyLoss()  # 多类交叉熵损失
best_model_path='/hpc2hdd/home/xingmu/MedSAM/checkpoints/unet_tumor_best.pth'
optimizer=optim.Adam(model.parameters(),lr=1e-3)
best_loss=float('inf')
epochs=200
train_losses=[]
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels[:, 0, :, :]  # 结果形状为 (8, 256, 256)
        labels = labels.long()
        optimizer.zero_grad()
        #labels[labels == 255] = 0
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算训练损失
        train_loss += loss.item()
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), best_model_path)

    # 打印结果
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}')

    # 存储训练损失
    train_losses.append(train_loss / len(train_loader))

torch.save(model.state_dict(), "/hpc2hdd/home/xingmu/MedSAM/checkpoints/unet_tumor_last.pth")
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('unet_tumor Training Loss Curve')
plt.legend()
plt.grid()
plt.savefig('unet_tumor_loss_curve.png')  # 保存图像到文件
print("Training loss curve saved as 'training_loss_curve.png'.")
