import torch
import torch.nn as nn
from utils import intersection_over_union

class yolov1_Loss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self,predictions,target):
        box_predictions = predictions.reshape(-1, self.S, self.S, (self.C + self.B*5) )
        # target = target.reshape(-1, self.S, self.S, (self.C + self.B*5) )
        # Stack the IoU tensors along a new dimension, then find the max
        iou_tensor = torch.stack([
        intersection_over_union(box_predictions[...,21:25], target[...,21:25], box_format="midpoint"),
        intersection_over_union(box_predictions[...,26:30], target[...,21:25], box_format="midpoint")
        ], dim=0)

        _, best_iou = torch.max(iou_tensor, dim=0)
        epsilon = 1e-6

        exists_box = target[...,20:21]

        best_box_predictions = exists_box * ((1-best_iou)*box_predictions[...,21:25] + best_iou*box_predictions[...,26:30])
        best_box_predictions[...,2:] = torch.sign(best_box_predictions[...,2:]) * torch.sqrt(torch.abs(best_box_predictions[...,2:] + epsilon))
        box_targets = exists_box * target[...,21:25] 
        box_targets[...,2:4] = torch.sign(box_targets[...,2:4]) *torch.sqrt(torch.abs(box_targets[...,2:4] + epsilon))

        box_loss = self.mse(
            torch.flatten(best_box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        object_loss = self.mse(
            torch.flatten(exists_box*((1-best_iou)*box_predictions[...,20:21] + best_iou*box_predictions[...,25:26]), end_dim=-2),
            torch.flatten(exists_box, end_dim=-2)
        )

        no_object_loss=self.mse(
            torch.flatten((1-exists_box) * box_predictions[...,20:21], end_dim=-2),
            torch.flatten((1-exists_box) * target[...,20:21], end_dim=-2)
        ) + self.mse(
            torch.flatten((1-exists_box) * box_predictions[...,25:26], end_dim=-2),
            torch.flatten((1-exists_box) * target[...,20:21], end_dim=-2)
        )
        class_loss = self.mse(
            torch.flatten(exists_box * box_predictions[...,:20], end_dim=-2),
            torch.flatten(exists_box * target[...,:20], end_dim=-2)
        )
        loss = (self.lambda_coord*box_loss + object_loss + self.lambda_noobj*no_object_loss + class_loss)
        return loss
