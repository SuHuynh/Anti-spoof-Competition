import torch
from torch import Tensor
import torch.nn as nn
from losses.focal_loss import FocalLoss

class First_Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.focal_loss = FocalLoss()
        self.sofmax_loss = nn.CrossEntropyLoss()
        self.BCE_loss = nn.BCEWithLogitsLoss()

    def forward(self, atr_pred, spoof_type_pred, illum_pred, env_pred, spoof_pred, atr_label, spoof_type_label, illum_label, env_label, spoof_label):

        spoof_label = spoof_label.float()
        spoof_pred = spoof_pred.squeeze()
        spoof_loss = self.focal_loss(spoof_pred, spoof_label)

        # import ipdb;ipdb.set_trace()
        atr_loss = self.BCE_loss(atr_pred.float(), atr_label.float())

        spoof_type_loss = self.sofmax_loss(spoof_type_pred, spoof_type_label)
        # import ipdb;ipdb.set_trace()
        illum_loss = self.sofmax_loss(illum_pred, illum_label)

        # env_loss = self.sofmax_loss(env_pred, env_label)

        loss = spoof_loss + 0.1*atr_loss + 0.1*spoof_type_loss + 0.01*illum_loss

        return loss, spoof_loss, 0.1*atr_loss, 0.1*spoof_type_loss, 0.01*illum_loss

class Recur_Loss(nn.Module):

    def __init__(self):
        super().__init__()

        self.focal_loss = FocalLoss()

    def forward(self, spoof_pred, spoof_label):

        spoof_label = spoof_label.float()
        spoof_pred = spoof_pred.squeeze()
        spoof_loss = self.focal_loss(spoof_pred, spoof_label)

        return spoof_loss