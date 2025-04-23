import torch
import torch.nn as nn
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomedModel(nn.Module):
    def __init__(self, anchor_generator, roi_pooler, num_classes:int, pretrained = True):
        super(CustomedModel, self).__init__()
        # trainable_backbone_layers
        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
            )
        
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features = model.roi_heads.mask_predictor.conv_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features, hidden_layer, num_classes)
        
        self.model = model

    def forward(self, images, targets = None):
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)