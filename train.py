import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import yolov1
import torch.optim as optim
from data import CustomVocDataset
import torchvision.transforms.functional as FT
from tqdm import tqdm
from utils import (intersection_over_union,
                   nms,
                   mean_average_precision,
                   plot_image,
                   get_bboxes,
                   convert_cellboxes,
                   cellboxes_to_boxes,
                   save_checkpoint,
                   load_checkpoint)
from loss import yolov1_Loss
import os

seed = 123
torch.manual_seed(seed)


LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 16 # 64 in original paper 
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = 'C:\Desktop\OpenCV\yolo_implementation\my_checkpoint.pth'
IMG_DIR = os.path.join('.','yolo_implementation','data_files','images')
LABEL_DIR = os.path.join('.','yolo_implementation','data_files','labels')

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes
    
transform = Compose([transforms.Resize((448,448)), transforms.ToTensor(),])


def train_func(train_loader, model, optimizer, loss_func):
    loop = tqdm(train_loader, leave =True)
    mean_loss =[]

    for batch_idx, (x,labels) in enumerate(loop):

        output_pred = model(x)

        loss = loss_func(output_pred,labels)

        optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()

        mean_loss.append(loss.item())
        loop.set_postfix(loss=loss.item())
    
    print(f'Mean Loss of 1 epoch = {sum(mean_loss)/len(mean_loss)}')

def main():
    model = yolov1(split_size=7, num_boxes=2, num_classes=20) #as i do not have gpu so cannot mount on it
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_func = yolov1_Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = CustomVocDataset('C:\Desktop\OpenCV\yolo_implementation\data_files\8examples.csv', img_dir=IMG_DIR, label_dir=LABEL_DIR, transform=transform)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle = True,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle = True,
        drop_last=False
    )

    for epoch in range(EPOCHS):
        for x,y in train_loader:
            for idx in range(8):
                # predictions = model(x)
                bboxes = cellboxes_to_boxes(model(x))
                
                bboxes = nms(bboxes[idx], iou_threshhold=0.2, pred_threshhold=0.2, box_format="midpoint")
                print(bboxes)
                plot_image(x[idx].permute(1,2,0).to("cpu"),bboxes)

            import sys
            sys.exit()




        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        print(f' training map: {mean_avg_prec}')

        if(epoch == 99):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            import time
            time.sleep(10)

        train_func(train_loader, model, optimizer, loss_func)

    


if __name__ == "__main__":
    main()