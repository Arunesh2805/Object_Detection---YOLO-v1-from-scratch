import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomVocDataset(Dataset):

    def __init__(self, csv_file, img_dir, label_dir, S = 7, B = 2, C = 20, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.C =C
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):

        label_path = os.path.join("C:/Desktop/OpenCV/yolo_implementation/data_files/labels", self.annotations.iloc[index, 1])
        boxes = []

        # print(f"Trying to open: {label_path}")

        with open(label_path) as f:
            
            for label in f.readlines():
                class_label, x, y, width, height = [ float(x) if float(x) != int(float(x)) else int(x)
                                                    for x in label.strip().split() ]
                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join("C:/Desktop/OpenCV/yolo_implementation/data_files/images", self.annotations.iloc[index, 0])
        img = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            img, boxes = self.transform(img, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = ((self.S * x) - j), ((self.S * y) - i)
            width_cell, height_cell = (self.S * width), (self.S * height)

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                label_matrix[i, j,21:25] = torch.tensor([ x_cell, y_cell, width_cell, height_cell ])
                label_matrix[i, j,class_label] = 1
        return img, label_matrix