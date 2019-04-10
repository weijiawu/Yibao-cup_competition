#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
from PIL import Image
from util.config import config as cfg
import numpy as np


def compute_mIOU_score(y_true, y_pred, smooth=1):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_f, y_pred_f = y_true.flatten().astype(int), y_pred.flatten().astype(int)
    I = (y_true_f & y_pred_f).sum()
    U = (y_true_f | y_pred_f).sum()
    return (I + smooth) / (U + smooth)

def sigmoid(x):
  return 1. / (1 + np.exp(-x))
def evaluation(image,output,one_mask, two_mask, all_mask,all_iou):
    for idx in range(image.size(0)):
        one_pred = output[idx, 0].data.cpu().numpy()
        two_pred = output[idx, 1].data.cpu().numpy()

        one_pred_ = sigmoid(one_pred)
        two_pred_ = sigmoid(two_pred)

        one_pred_, two_pred_ = np.where(one_pred_ > 0.5, 1, 0), np.where(two_pred_ > 0.5, 1, 0)

        img_show = image[idx].permute(1, 2, 0).cpu().numpy()

        one_mask_ = one_mask[idx].cpu().numpy()
        two_mask_ = two_mask[idx].cpu().numpy()

        overlap_mask = one_mask_ * two_mask_
        overlap_pred = one_pred_ * two_pred_


        overlap_iou = compute_mIOU_score(overlap_mask,overlap_pred)
        one_iou = compute_mIOU_score((one_mask_ - overlap_mask),(one_pred_ - overlap_pred))
        two_iou = compute_mIOU_score((two_mask_ - overlap_mask), (two_pred_ - overlap_pred))
        all_iou += (overlap_iou * 0.8 + one_iou * 0.1 + two_iou * 0.1)
        
    return all_iou


    