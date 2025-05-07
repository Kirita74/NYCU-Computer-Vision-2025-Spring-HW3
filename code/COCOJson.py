import os 
import matplotlib.pyplot as plt
import cv2
import skimage.io as sio
from code.utils import encode_mask,decode_maskobj,read_maskfile
from scipy import ndimage
import numpy as np
import torch
import re
import json
dirs = os.listdir("data/train")
annotation_infos = []
annotation_id = 1

def mask_instance(image_id, mask_label, mask_path):
    global annotation_id
    mask = sio.imread(mask_path) 
    binary_mask = (mask > 0)

    ys, xs = np.where(binary_mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    labeled, num_objs = ndimage.label(binary_mask)

    for i in range(1,num_objs+1): #labeled == 0 will output all mask instance
        mask_instace = (labeled == i)
        print(mask_instace.shape)
        exit()
        #bbox
        ys, xs = np.where(mask_instace)
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        if xmax <= xmin or ymax <= ymin:
            continue

        anno_info = {
            "image_id" : image_id,
            "bbox":[int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)],
            "category_id":mask_label,
            "segmentation":encode_mask(mask_instace)
        }

        annotation_infos.append(anno_info)
        annotation_id += 1

if __name__ == "__main__":

    img_infos = []

    for idx,subdir in enumerate(dirs):
        image_id = idx + 1

        for filename in os.listdir(f"data\\train\\{dirs[idx]}"):
            if(filename == "image.tif"):
                image_path = os.path.join(f"data\\train\\{dirs[idx]}",filename)
            else:
                mask_label = re.findall(r'\d+', filename) [0]
                #mask_pairs.append((mask_label,os.path.join(f"data\\train\\{dirs[idx]}",filename)))
                mask_instance(image_id, mask_label,os.path.join(f"data\\train\\{dirs[idx]}",filename) )
        img_info = {
            "id" : image_id,
            "filename": subdir
        }
        img_infos.append(img_info)

    categories=[]
    
    for i in range(4):
        categories.append({
            "id":i+1,
            "name":i+11
        })
    
    jsonDict = {
        "images": img_infos,
        "annotations": annotation_infos,
        "categories": categories
    }
   
    with open("sample.json", "w") as outfile:
        json.dump(jsonDict, outfile)




