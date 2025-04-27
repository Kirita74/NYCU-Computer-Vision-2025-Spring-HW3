import gc
import torch
import json
from tqdm import tqdm
from dataset import CustomedDataset, TestDataset
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
anchor_generator = AnchorGenerator(
    sizes=((4,), (8,), (16,), (32,), (64,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=['0', '1', '2', '3'],
    output_size=7,
    sampling_ratio=2
)

def load_Data():
    full_dataset = CustomedDataset(TRAIN_PATH)
    train_size = int(0.9 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset,[train_size, valid_size])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    return train_loader,valid_loader

def train():
    tensorBdWriter = SummaryWriter(LOG_DIR)
    train_loader, valid_loader= load_Data()
    model = CustomedModel(num_classes=NUM_CLASSES + 1, anchor_generator=anchor_generator, roi_pooler=roi_pooler, pretrained=True)
    model.load_pretrained_weight(pretrained_weight_path="model2.pth",weight_only=True,device=DEVICE)
    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=OPTIMIZER_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    val_metric = MeanAveragePrecision()
    best_map = 0.0

    for epoch in range(EPOCHS):
        model.train()
        gc.collect()
        epoch_loss = 0.0
    
        for images, targets in tqdm(train_loader,desc=f"Epoch {epoch+1}"):
            #transfer data to GPU
            images_gpu = [img.to(DEVICE) for img in images]
            targets_gpu = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]
            loss_dict = model(images_gpu, targets_gpu)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch + 201}] Loss: {avg_epoch_loss:.4f}")

        tensorBdWriter.add_scalar("Loss/train", avg_epoch_loss, epoch+200)
        tensorBdWriter.add_scalar("Loss/Learning rate", scheduler.get_last_lr()[0], epoch+200)

        model.eval() #change model to eval mode
        val_metric.reset()
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [img.to(DEVICE) for img in images]

                predicteds = model(images)
                predicteds = [{k: v.cpu() for k, v in p.items()} for p in predicteds]

                val_metric.update(preds=predicteds,target=targets)
        metrics = val_metric.compute()
        print(f"Valid: [Epoch {epoch+201}] mAP: {metrics['map']:.4f}, mAP50: {metrics['map_50']:.4f} mAP75: {metrics['map_75']:.4f}")
        tensorBdWriter.add_scalar('mAP/val', metrics['map'], epoch+200)
        tensorBdWriter.add_scalar('mAP50/val', metrics['map_50'], epoch+200)
        tensorBdWriter.add_scalar("mAP75/val", metrics['map_75'], epoch+200)
        
        if metrics["map"] > best_map:
            torch.save(model.state_dict(), "model3.pth")

    # TODO: vaild and test function (adjust batch size)

def test():
    test_dataset = TestDataset(test_path=TEST_PATH)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

    model = CustomedModel(anchor_generator=anchor_generator, roi_pooler=roi_pooler, num_classes= 4+1, pretrained=False)
    model.load_pretrained_weight("model3.pth")
    model.to(DEVICE)
    model.eval()

    results = []
    image_name_to_ids = []
    for file_id, file_name, images in tqdm(test_loader,desc = "Testing"):
        file_id = file_id.item()
        images = [img.to(DEVICE) for img in images]
        predicted = model(images)
        predicted = [{k: v.cpu() for k, v in p.items()} for p in predicted]
        image_size = []
        for p in predicted:
            bboxes = p["boxes"]
            labels = p["labels"]
            scores = p["scores"]
            masks = p["masks"]
            image_size = masks[0].shape[1:]
            for idx in range(bboxes.shape[0]):
    
                bbox = bboxes[idx].tolist()
                label = labels[idx].item()
                score = scores[idx].item()
                mask = masks[idx].detach().numpy()
                rle_mask = encode_mask(mask.squeeze(axis = 0))
                
                result = {
                    "image_id":file_id,
                    "bbox":bbox,
                    "score":score,
                    "category_id":label,
                    "segmentation":rle_mask
                }
                results.append(result)
  
        image_name_to_ids.append({
            "file_name":file_name[0],
            "id":file_id,
            "height":image_size[1],
            "width":image_size[0]
        })

    with open("test_image_name_to_ids.json","w") as outfile:
        json.dump(image_name_to_ids, outfile)

    with open("test.json", "w") as outfile:
        json.dump(results, outfile)


if __name__=="__main__":
    TRAIN_PATH = "data\\train"
    TEST_PATH = "data\\test_release"
    BATCH_SIZE = 1
    EPOCHS = 100
    LEARNING_RATE = 5e-5
    OPTIMIZER_WEIGHT_DECAY = 1e-5
    LOG_DIR = "logs/log1"
    test()