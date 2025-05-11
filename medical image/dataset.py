import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Change the file extension for the label
        label_name = os.path.splitext(image_name)[0] + '.jpg'  # Assuming label is .png
        label_path = os.path.join(self.label_dir, label_name)

        # Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path) 
        label=np.array(label)
        label[label<128]=0
        label[label>128]=1
        label = cv2.resize(label, (512,512),interpolation=cv2.INTER_NEAREST)
        label=torch.tensor(label)
        label=label.unsqueeze(0)
        # Resize both image and label
        #image = image.resize(self.resize, Image.BILINEAR)
        #label = label.resize(self.resize, Image.NEAREST)
        
        #label = Image.fromarray(label.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
            #label =self.transform(label)
        return image, label

