import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, resize=(384, 384)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)
        self.resize = resize

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # Change the file extension for the label
        label_name = os.path.splitext(image_name)[0] + '.png'  # Assuming label is .png
        label_path = os.path.join(self.label_dir, label_name)

        # Load image and label
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        # Resize both image and label
        image = image.resize(self.resize, Image.BILINEAR)
        label = label.resize(self.resize, Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        # Convert label to Tensor
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.long)
        if label.dim() == 2:  # If label is 2D (H, W), add a channel dimension
            label = label.unsqueeze(0)  # Shape becomes (1, H, W)

        return image, label

