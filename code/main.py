import torch
from tqdm import tqdm
from dataset import CustomedDataset
from torch.utils.data.dataloader import DataLoader
from model import CustomedModel
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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
    train_dataset = CustomedDataset(TRAIN_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    return train_dataset, train_loader

def train():
    train_dataset, train_loader = load_Data()
    model = CustomedModel(num_classes=NUM_CLASSES + 1, anchor_generator=anchor_generator, roi_pooler=roi_pooler, pretrained=True)
    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=OPTIMIZER_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    val_metric = MeanAveragePrecision()

    for epoch in range(EPOCHS):
        model.train()
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

    # TODO: vaild and test function (adjust batch size)
if __name__=="__main__":
    TRAIN_PATH = "data\\train"
    BATCH_SIZE = 2
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    OPTIMIZER_WEIGHT_DECAY = 1e-5
    train()