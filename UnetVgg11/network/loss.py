import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BCELoss2d(nn.Module):
    """
    Binary Cross Entropy loss function
    """
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1)
        return self.bce_loss(logits_flat, labels_flat)


class WeightedBCELoss2d(nn.Module):
    def __init__(self):
        super(WeightedBCELoss2d, self).__init__()

    def forward(self, logits, labels, weights):
        w = weights.view(-1)
        logits = logits.view(-1)
        gt = labels.view(-1)
        # http://geek.csdn.net/news/detail/126833
        loss = logits.clamp(min=0) - logits * gt + torch.log(1 + torch.exp(-logits.abs()))
        loss = loss * w
        loss = loss.sum() / w.sum()
        return loss


class WeightedSoftDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, logits, labels, weights):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        w = weights.view(num, -1)
        w2 = w * w
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / (
            (w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score

class TextLoss(nn.Module):

    def __init__(self, is_weight=True, is_log_dice=False):
        super().__init__()
        self.is_weight = is_weight
        self.is_log_dice = is_log_dice

        self.weighted_bce = WeightedBCELoss2d()
        self.soft_weighted_dice = WeightedSoftDiceLoss()

        self.bce = BCELoss2d()
        self.soft_dice = SoftDiceLoss()


    def forward(self,output, one_mask, two_mask, all_mask,one_mask_weight, two_mask_weight,box_mask):
        """
        calculate textsnake loss
        :param input: (Variable), network predict, (BS, 7, H, W)
        :param tr_mask: (Variable), TR target, (BS, H, W)
        :param tcl_mask: (Variable), TCL target, (BS, H, W)
        :param sin_map: (Variable), sin target, (BS, H, W)
        :param cos_map: (Variable), cos target, (BS, H, W)
        :param radii_map: (Variable), radius target, (BS, H, W)
        :param train_mask: (Variable), training mask, (BS, H, W)
        :return: loss_tr, loss_tcl, loss_radii, loss_sin, loss_cos
        """
        one_pred = output[:, 0].contiguous()
        two_pred = output[:, 1].contiguous()
        box_pred = output[:, 2].contiguous()

        one_mask = one_mask.contiguous()
        two_mask = two_mask.contiguous()
        box_mask = box_mask.contiguous()

        one_mask_weight = one_mask_weight.contiguous()
        two_mask_weight = two_mask_weight.contiguous()

        bce_loss = self.weighted_bce(one_pred, one_mask,one_mask_weight) + self.weighted_bce(two_pred, two_mask,two_mask_weight)
        dice_loss = self.soft_dice(one_pred, one_mask) + self.soft_dice(two_pred, two_mask)

        loss_box = F.smooth_l1_loss(box_pred, box_mask.float())

        return bce_loss,dice_loss,loss_box