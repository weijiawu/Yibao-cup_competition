#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import  numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from network.vgg_unet import UnetVgg11
from util.augmentation import BaseTransform
from dataset.total_text import TotalText_test
from util.option import BaseOptions
from util.config import config as cfg, update_config, print_config
from torch.autograd import Variable
img_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/test/'
output_path = '/home/weijia.wu/workspace/Kaggle/Word_segementation/Carvana_Image/UnetVgg11/Demo/result/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 2
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)
model_path = ''


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])

import time
def sigmoid(x):
  return 1. / (1 + np.exp(-x))
def main():
    data_Ids = [f.split('.')[0] for f in os.listdir(img_path)]
    image_datasets = TotalText_test(data_Ids=list(data_Ids), imgs_path=img_path)

    image_dataloader = data.DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=15)

    import cv2
    model = UnetVgg11()
    model.cuda()
    model = nn.DataParallel(model)
    # model = nn.DataParallel(model)

    model_path = '/home/weijia.wu/workspace/Kaggle/Word_segementation/Carvana_Image/UnetVgg11/save_model/test/textsegementation_best.pth'
    load_model(model, model_path)

    print('Start testing TextSnake.')
    start = time.time()
    model.eval()
    for i, (image,meta) in enumerate(image_dataloader):
        image = Variable(image.cuda())
        output = model(image)
        for idx in range(image.size(0)):

            one_pred = output[idx, 0].data.cpu().numpy()
            two_pred = output[idx, 1].data.cpu().numpy()

            one_pred_ = sigmoid(one_pred)
            two_pred_ = sigmoid(two_pred)

            one_pred_, two_pred_ = np.where(one_pred_ > 0.5, 1, 0), np.where(two_pred_ > 0.5, 1, 0)

            img_show = image[idx].permute(1, 2, 0).cpu().numpy()
            result = one_pred_ * 60 + two_pred_ * 120

            path = os.path.join(output_path,meta['image_id'][idx]+'.png')
            print(path)
            # cv2.imwrite(path, result)
        break
    print(time.time()-start)
if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()