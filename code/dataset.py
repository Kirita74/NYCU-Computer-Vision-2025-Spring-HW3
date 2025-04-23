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

class CustomedDataset(Dataset):
    def __init__(self, path, transform = None):
        super().__init__()
        self.path = path
        self.data_subdirs = os.listdir(path)

        if (transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456, 0.406],
                                     std=[0.229, 0.225, 0.225])
            ])
        else:
            self.transform = transform
    
    def __getitem__(self, idx):
        subdir = os.path.join(self.path,self.data_subdirs[idx])
        
        mask_pairs= []
        for file in os.listdir(subdir):
            if(file == "image.tif"):
                image_path = os.path.join(subdir,file)
            else:
                mask_label = re.findall(r'\d+', file) [0]
                mask_pairs.append((int(mask_label), os.path.join(subdir,file)))
                
        image = Image.open(image_path).convert("RGB")
      
        image = self.transform(image)

        binary_masks = []
        boxes = []
        labels = []
        for mask_label,mask_path in mask_pairs:
            binary_mask, mask_boxes, mask_labels = self.getMaskInstace(mask_label,mask_path)
            binary_masks += binary_mask
            boxes += mask_boxes
            labels += mask_labels 

        binary_masks_np = np.array(binary_masks)
        
        target = {
            "image_id" : torch.tensor([idx]),
            "boxes" : torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks" : torch.from_numpy(binary_masks_np).to(torch.uint8)
        }

        return image, target

    def __len__(self):
        return len(self.data_subdirs)
    
    def getMaskInstace(self, mask_label, mask_path):
        mask = sio.imread(mask_path)
        binary_mask = (mask > 0)

        labeled, num_objs = ndimage.label(binary_mask)

        masks = []
        labels = []
        boxes = []

        for i in range(1,num_objs+1): #labeled == 0 will output all mask instance
            mask_instace = (labeled == i)

            #bbox
            ys, xs = np.where(mask_instace)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            
            # Debug
            if xmax <= xmin or ymax <= ymin:
                continue
            
            masks.append(np.array(mask_instace, dtype=np.uint8))
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(mask_label)

        return masks, boxes, labels


       


    