import gc
import torch
import json
from tqdm import tqdm
from dataset import TestDataset, getDatasets
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from model import CustomedModel
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
from utils import encode_mask
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
anchor_generator = AnchorGenerator(
    sizes=((4,), (8,), (16,), (32,), (64,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1','2','3'],
    output_size=7,
    sampling_ratio=2
)

mask_roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=14,
    sampling_ratio=4
)

def load_Data():
    train_dataset, valid_dataset = getDatasets(train_path=TRAIN_PATH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    return train_loader,valid_loader

def train():
    tensorBdWriter = SummaryWriter(LOG_DIR)
    train_loader, valid_loader= load_Data()
    model = CustomedModel(num_classes=NUM_CLASSES + 1, anchor_generator=anchor_generator, roi_pooler=roi_pooler, mask_roi_pooler=mask_roi_pooler, pretrained=True)
    # model.load_pretrained_weight(pretrained_weight_path=PRETRAINED_WEIGHT_PATH, weight_only=True,device=DEVICE)
    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=OPTIMIZER_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(device='cuda')
    val_metric = MeanAveragePrecision()
    best_map = 0.0

    for epoch in range(EPOCHS):
        model.train()
        gc.collect()
        epoch_loss = 0.0
    
        for images, targets in tqdm(train_loader,desc=f"Epoch {epoch+1 + E}"):
            #transfer data to GPU
            images_gpu = [img.to(DEVICE) for img in images]
            targets_gpu = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', cache_enabled=True):
                loss_dict = model(images_gpu, targets_gpu)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            epoch_loss += loss.item()

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch + 1 + E}] Loss: {avg_epoch_loss:.4f}")

        tensorBdWriter.add_scalar("Loss/train", avg_epoch_loss, epoch+ E)
        tensorBdWriter.add_scalar("Loss/Learning rate", scheduler.get_last_lr()[0], epoch + E)

        model.eval() #change model to eval mode
        val_metric.reset()
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(DEVICE) for img in images]

                predicteds = model(images)
                predicteds = [{k: v.cpu() for k, v in p.items()} for p in predicteds]

                processed_pred = []
                
                for pred in predicteds:
                    mask = torch.sigmoid(pred["masks"])
                    mask = mask.detach().numpy()
                    mask = mask > 0.5

                    if(mask.sum() == 0):
                        continue

                    processed_pred.append(pred)
                if(len(processed_pred) == 0):
                    continue
                val_metric.update(preds=processed_pred, target=targets)

        metrics = val_metric.compute()
        print(f"Valid: [Epoch {epoch+1}] mAP: {metrics['map']:.4f}, mAP50: {metrics['map_50']:.4f} mAP75: {metrics['map_75']:.4f}")
        tensorBdWriter.add_scalar('mAP/val', metrics['map'], epoch + E)
        tensorBdWriter.add_scalar('mAP50/val', metrics['map_50'], epoch + E)
        tensorBdWriter.add_scalar("mAP75/val", metrics['map_75'], epoch + E)
        
        with open(f"{LOG_DIR}\\mAP record.txt","a+") as mApwrt:
            mApwrt.writelines(f"[Epoch {epoch + 1 + E}]\n")

            for k,v in metrics.items():
                if(k == "classes"):
                    continue
                if isinstance(v, torch.Tensor):
                    v = float(v)
                mApwrt.writelines(f"{k}:{v:.4f}\n")

        if metrics["map"] > best_map:
            torch.save(model.state_dict(), PRETRAINED_WEIGHT_PATH)

def test():
    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    model = CustomedModel(num_classes= NUM_CLASSES + 1, anchor_generator=anchor_generator, roi_pooler=roi_pooler, mask_roi_pooler=mask_roi_pooler, pretrained=False)
    model.load_pretrained_weight(PRETRAINED_WEIGHT_PATH)
    model.to(DEVICE)
    model.eval()

    results = []
    
    for file_id, file_size,images in tqdm(test_loader,desc = "Testing"):
        file_id = file_id.item()
        images = [img.to(DEVICE) for img in images]
        predicted = model(images)
        predicted = [{k: v.cpu() for k, v in p.items()} for p in predicted]
        for p in predicted:
            bboxes = p["boxes"]
            labels = p["labels"]
            scores = p["scores"]
            masks = p["masks"]

            for idx in range(bboxes.shape[0]):
                bbox = bboxes[idx].tolist()
                label = labels[idx].item()
                score = scores[idx].item()
                mask = torch.sigmoid(masks[idx])
                mask = mask.detach().numpy()
                mask = mask > 0.5

                if(mask.sum() == 0):
                    continue

                x_min, y_min, x_max,y_max = bbox
                if(x_min >= file_size[1] or y_min >= file_size[0]):
                    continue
                
                bbox = [x_min, y_min,x_max-x_min, y_max-y_min]
                rle_mask = encode_mask(mask.squeeze(axis = 0))
                result = {
                    "image_id":file_id,
                    "bbox":bbox,
                    "score":score,
                    "category_id":label,
                    "segmentation":rle_mask
                }
                results.append(result)
                # print(result)

    with open("test-results.json", "w", encoding='utf-8') as outfile:
        json.dump(results, outfile)


if __name__=="__main__":
    TRAIN_PATH = "data\\train"
    BATCH_SIZE = 1
    EPOCHS = 60
    LEARNING_RATE = 1e-4
    OPTIMIZER_WEIGHT_DECAY = 5e-5
    LOG_DIR = "logs/log17"
    #調整RPN_HEAD.conv_depth = 3, MaskPredictor變3層 model:test_1.pth
    PRETRAINED_WEIGHT_PATH = "weights\\convDepth5_hiddenLayer512_aug_resnextv2.pth"
    MASK_THERSHOLD = 0.5
    E = 0
    train()
    # test()