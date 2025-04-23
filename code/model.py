import torch
import torch.nn as nn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn,MaskRCNN_ResNet50_FPN_Weights, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead
import os
class CustomedModel(nn.Module):
    def __init__(self, anchor_generator, roi_pooler, num_classes:int, pretrained = True):
        super(CustomedModel, self).__init__()
        # trainable_backbone_layers
        model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            rpn_anchor_generator = anchor_generator,
            box_roi_pool = roi_pooler
            )
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
    
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.roi_heads.nms_thresh = 0.2
        model.roi_heads.score_thresh = 0.5
        model.roi_heads.positive_fraction = 0.25
        
        #Mask Head跟RPN head可以改
        in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features, hidden_layer, num_classes)
        
        # model.rpn_anchor_generator = anchor_generator
        # model.box_roi_pool = roi_pooler
        self.model = model

    def forward(self, images, targets = None):
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
        
    def load_pretrained_weight(self, pretrained_weight_path:str, device):
        if(os.path.exists(pretrained_weight_path)):
            self.model.load_state_dict(torch.load(pretrained_weight_path, map_location=device))
        else:
            print("pretrainde weight not exist.")
    
    def save_model(self, save_model_path:str):
        torch.save(self.model.state_dict(),save_model_path)