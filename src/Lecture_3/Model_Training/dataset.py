import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np

class BeeDataset(Dataset):
    """Custom Dataset for Loading Bee Images and Masks"""

    def __init__(self, imgs_dir, masks_dir):
        """
        Args:
            imgs_dir (str): directory with images
            masks_dir (str): directory with masks
            transform (callable, optional): optional transform
        """
        self.imgs  = sorted([f for f in os.listdir(imgs_dir)  if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.jpg')])

        assert len(self.imgs) == len(self.masks), "Numbers of images and masks do not match!"

        self.imgs_dir  = imgs_dir
        self.masks_dir = masks_dir
        self.image_size = (256, 256)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
            transforms.Resize(self.image_size)
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_file  = os.path.join(self.imgs_dir, self.imgs[idx]) 
        mask_file = os.path.join(self.masks_dir, self.masks[idx])

        img = Image.open(img_file)
        img = self.transform(img)
        
        mask = Image.open(mask_file).convert("L")  # convert to grayscale
        # Don't normalize the mask, just resize it
        mask = mask.resize(self.image_size, Image.BILINEAR) # apply bilinear interpolation to remove artifacts from resizing
        
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).float() # Each mask has 0 (background) and 255 (bee)
        mask = mask.unsqueeze(0) # Add channel dimension (C, H, W)
        
        # Boxes are [xmin, ymin, xmax, ymax]
        pos = np.where(mask[0] == 1)
        xmin = np.min(pos[1]) 
        ymin = np.min(pos[0]) 
        xmax = np.max(pos[1]) 
        ymax = np.max(pos[0])

        boxes = torch.FloatTensor([
            [xmin, ymin, xmax, ymax],
        ])

        # Labels (1 for bee)
        labels = torch.ones((1,), dtype=torch.int64)

        # Masks
        masks = torch.FloatTensor(mask)

        # Image id
        image_id = torch.tensor([idx])

        # Area
        area = (xmax - xmin) * (ymax - ymin)

        # Iscrowd (always 0)
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": torch.FloatTensor([area]),
            "iscrowd": iscrowd,
        }

        return img, target
    
def collate_fn(batch): # needed for Dataloader
    return tuple(zip(*batch))