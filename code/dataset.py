from torch.utils.data import Dataset
from torchvision import transforms 
from PIL import Image
import os
from pycocotools import mask as mask
import skimage.io as sio
import numpy as np
import cv2
from scipy import ndimage
import torch
import re
from utils import encode_mask, decode_maskobj
import json

class CustomedDataset(Dataset):
    def __init__(self, filePath, transform = None):
        super().__init__()
        json_data = json.load(open(f"{filePath}\\sample.json","r"))
        self.images = json_data["images"]
        self.annotations = json_data["annotations"]
        self.categories = json_data["categories"]
        self.folder = filePath

        if (transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456, 0.406],
                                     std=[0.229, 0.225, 0.225])
            ])
        else:
            self.transform = transform
    
    def __getitem__(self, idx):
        filename = self.images[idx]["filename"] 
        
        img = Image.open(os.path.join(self.folder, filename,"image.tif")).convert("RGB")

        file_id = self.images[idx]["id"]

        image_annotations = [anno for anno in self.annotations if anno["image_id"] == file_id]
        
        boxes = []
        labels = []
        binary_masks = []

        for anno in image_annotations:
            x, y, bw, bh = anno["bbox"]
            x_min = x
            y_min = y
            x_max = x + bw
            y_max = y + bh

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(anno["category_id"]))
            binary_masks.append(decode_maskobj(anno["segmentation"]).astype(np.uint8))

        image = self.transform(img)

        binary_masks_np = np.array(binary_masks)
        target = {
            "image_id" : torch.tensor([file_id]),
            "boxes" : torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks" : torch.from_numpy(binary_masks_np)
        }
        return image, target

    def __len__(self):
        return len(self.images)
    
class TestDataset(Dataset):
    def __init__(self, test_path):
        super().__init__()
        self.root = test_path
        self.image_paths = os.listdir(test_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456, 0.406],
                                std=[0.229, 0.225, 0.225])
        ])

    def __getitem__(self, index):
        file_name = self.image_paths[index].split(sep=".")[0]
        filepath = os.path.join(self.root, self.image_paths[index])
        img =  Image.open(filepath).convert("RGB")
        img_transformed = self.transform(img)

        return index + 1, file_name, img_transformed  

    def __len__(self):
        return len(self.image_paths)

 

       


    