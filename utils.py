import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import os

def intersection_over_union(bbox_pred, bbox_label, box_format='box'):
    
    bbox1 = bbox_pred  
    bbox2 = bbox_label
    if not torch.is_tensor(bbox1):
        bbox1 = torch.tensor(bbox1)
    if not torch.is_tensor(bbox2):
        bbox2 = torch.tensor(bbox2)
    
    # print("boxes_preds shape:", bbox_pred.shape)
    # print("boxes_labels shape:", bbox_label.shape)
    
    if box_format == 'midpoint':  
        box1_x1 = bbox1[..., 0:1] - bbox1[..., 2:3]/2
        box1_y1 = bbox1[..., 1:2] - bbox1[..., 3:4]/2
        box1_x2 = bbox1[..., 0:1] + bbox1[..., 2:3]/2
        box1_y2 = bbox1[..., 1:2] + bbox1[..., 3:4]/2
        box2_x1 = bbox2[..., 0:1] - bbox2[..., 2:3]/2
        box2_y1 = bbox2[..., 1:2] - bbox2[..., 3:4]/2
        box2_x2 = bbox2[..., 0:1] + bbox2[..., 2:3]/2
        box2_y2 = bbox2[..., 1:2] + bbox2[..., 3:4]/2
    elif box_format == 'box': 
        box1_x1 = bbox1[..., 0:1]
        box1_y1 = bbox1[..., 1:2]
        box1_x2 = bbox1[..., 2:3]
        box1_y2 = bbox1[..., 3:4]
        box2_x1 = bbox2[..., 0:1]
        box2_y1 = bbox2[..., 1:2]
        box2_x2 = bbox2[..., 2:3]
        box2_y2 = bbox2[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union = area1 + area2 - intersection + 1e-6
    iou = intersection / union
    return iou


def nms(bboxes,
        iou_threshhold,
        pred_threshhold,
        box_format = 'box'):
    
    # assert type(pred_bboxes) == list
    pred_bboxes = [box for box in bboxes if box[1] > pred_threshhold]
    pred_bboxes = sorted(pred_bboxes,key = lambda x : x[1], reverse=True)
    bbox_after_nms = []

    while pred_bboxes:
        correct_box = pred_bboxes.pop(0)

        pred_bboxes = [ box 
                       for box in pred_bboxes
                       if box[0] != correct_box[0] or intersection_over_union(box,correct_box,box_format=box_format) < iou_threshhold
                       ]
        
        bbox_after_nms.append(correct_box)

    return bbox_after_nms



def mean_average_precision(
        pred_boxes,true_boxes,iou_threshold,box_format='box',num_classes=20
):
    #pred_box = [img_index,object_class,probability,x1,y1,x2,y2]
    average_precisions=[]
    epsilon = 1e-6

    for object_class in range(num_classes):
        detected_class_bboxes = [ box
                                for box in pred_boxes
                                if box[1] == object_class ]
        true_class_bboxes = [box
                            for box in true_boxes
                            if box[1] == object_class]
        if len(true_class_bboxes) == 0:
            if len(detected_class_bboxes) == 0:
                # No predictions and no ground truth - perfect score
                average_precisions.append(1.0)
            else:
                # Predictions exist but no ground truth - worst score
                average_precisions.append(0.0)
            continue
        
        # Skip class if no predictions exist
        if len(detected_class_bboxes) == 0:
            average_precisions.append(0.0)
            continue


        amount_bbox = Counter([gt[0] for gt in true_class_bboxes])

        for key,val in amount_bbox.items():
            amount_bbox[key] = torch.zeros(val)
        
        detected_class_bboxes.sort(key=lambda x: x[2] ,reverse=True )
        TP = torch.zeros(len(detected_class_bboxes))
        FP = torch.zeros(len(detected_class_bboxes))
        total_true_bboxs = len(true_class_bboxes)

        for detection_index,detected_bbox in enumerate(detected_class_bboxes):
            true_class_bboxs_img = [ bbox
                                    for bbox in true_class_bboxes
                                    if bbox[0] == detected_bbox[0] ]
            best_iou=0
            best_tc_index =-1
            best_original_bbox = None 
            
            for tc_index,true_bbox in enumerate(true_class_bboxs_img):
                iou = intersection_over_union(detected_bbox,true_bbox,box_format)

                if iou > best_iou:
                    best_iou = iou
                    best_tc_index = tc_index
                    best_original_bbox = true_bbox

            if best_iou > iou_threshold and best_original_bbox is not None:

                img_gt_boxes = [gt for gt in true_class_bboxes if gt[0] == detected_bbox[0]]
                original_idx = img_gt_boxes.index(best_original_bbox)
                if amount_bbox[detected_bbox[0]][original_idx] == 0:
                    TP[detection_index] = 1
                    amount_bbox[detected_bbox[0]][original_idx] = 1
                else:
                    FP[detection_index] = 1
            else:
                FP[detection_index] = 1

        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        for i in range(len(TP_cumsum)):
            print(TP_cumsum[i])
        print("..............................................")
        for i in range(len(FP_cumsum)):
            print(FP_cumsum[i])
        precision = TP_cumsum/(TP_cumsum + FP_cumsum + epsilon)
        recall = TP_cumsum/(total_true_bboxs + epsilon)
        precision = torch.cat((torch.tensor([1.0],dtype=precision.dtype), precision), dim=0)
        recall = torch.cat((torch.tensor([0.0],dtype=recall.dtype), recall), dim=0)
        average_precisions.append(torch.trapz(precision, recall))

    return sum(average_precisions) / len(average_precisions)


def plot_image(img, bboxes):

    im = np.array(img)
    height, width, _ =im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    #bbox = [x, y, width, height]

    for bbox in bboxes:
        upper_left_x = bbox[2] - bbox[4]/2
        upper_left_y = bbox[3] - bbox[5]/2

        rect = patches.Rectangle((upper_left_x*width,upper_left_y*height),
                                 bbox[4]*width,
                                 bbox[5]*height,
                                 edgecolor = "b",
                                 linewidth = 2,
                                 facecolor = "none",
                                 )
        ax.add_patch(rect)

    plt.show()


def get_bboxes(loader, model, iou_threshold, threshold, pred_format = "cells", box_format = "midpoint", device = "cpu",S =7):
    all_pred_boxes =[]
    all_true_boxes =[]

    model.eval()
    train_idx = 0
    for batch_idx, (x,labels) in enumerate(loader):
        with torch.no_grad():
            predictions =model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_bboxes = nms(bboxes[idx],
                             iou_threshhold=iou_threshold,
                             pred_threshhold=threshold,
                             box_format=box_format)
            
            for nms_box in nms_bboxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for true_box in true_bboxes:
                # print(len(true_box))
                # print(len(true_box[0]))
                for x in range(S*S):
                    if(true_box[x][1] > threshold):
                        all_true_boxes.append([train_idx] + true_box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size,S,S,30)
    bbox1 = predictions[...,21:25]
    bbox2 = predictions[...,26:30]

    scores = torch.cat((predictions[...,20].unsqueeze(0), predictions[...,25].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bbox1*(1 - best_box) + bbox2*best_box
    cell_indices = torch.arange(S).repeat(batch_size,S,1).unsqueeze(-1)
    # print(best_boxes.shape)
    # print(cell_indices.shape)
    absolute_x = (best_boxes[...,0].unsqueeze(-1) + cell_indices) / S
    absolute_y = (best_boxes[...,1].unsqueeze(-1)  + cell_indices.permute(0,2,1,3))/S
    absolute_width = best_boxes[...,2].unsqueeze(-1) /S
    absolute_height = best_boxes[...,3].unsqueeze(-1) /S

    absolute_bboxes = torch.cat((absolute_x, absolute_y, absolute_width, absolute_height), dim=-1)
    predicted_class = predictions[...,:20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[...,20], predictions[...,25]).unsqueeze(-1)

    converted_predictions = torch.cat((predicted_class, best_confidence, absolute_bboxes), dim=-1)

    return converted_predictions

def cellboxes_to_boxes(out, S=7):
    converted_preditions = convert_cellboxes(out, S).reshape(out.shape[0], S*S, -1)
    converted_preditions[...,0] = converted_preditions[...,0].long()
    all_bboxes =[]

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_preditions[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename = 'C:\Desktop\OpenCV\yolo_implementation\my_checkpoint.pth'):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=>Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


