from torch.utils.data import Dataset
from torchvision import transforms 
from PIL import Image
import os
from pycocotools import mask as mask
import skimage.io as sio
import numpy as np
import cv2
from scipy import ndimage

class CustomedDataset(Dataset):
    def __init__(self, path, transform = None):
        super().__init__()
        self.path = path
        self.data_subdirs = os.listdir(path)

        if (transform == None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456, 0.406],
                                     std=[[0.229, 0.225, 0.225]])
            ])
        else:
            self.transform = transform
    
    def __getitem__(self, idx):
        subdir = self.data_subdirs[idx]
        
        mask_paths = []
        for file in os.listdir(subdir):
            if(file == "image.tif"):
                image_path = os.path.join(self.path,subdir,file)
        image = cv2.imread(str(image_path))

        for idx,mask_path in enumerate(mask_paths):
            mask = sio(idx, mask_path)


    def getMaskInstace(self, mask_label, mask_path):
        binary_mask = (mask > 0)

        labeled, num_objs = ndimage.label(binary_mask)

        labels = []
        boxes = []

        for i in range(1,num_objs+1): #labeled == 0 will output all mask instance
            mask_instace = (labeled == i)

            #bbox
            ys, xs = np.where(mask_instace)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            
            boxes.append([xmin,xmax,ymin,ymax])
            labels.append(mask_label)

                        
        target  = {
            "boxes" : torch.tensor(boxes, dtype=torch.float32),
            "labels" : torch.tensor(labels, dtype=torch.float32),
            "image_id" : 1
        }
        


       


    