#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import os
import  numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from modeling.deeplab import DeepLab
from util.augmentation import BaseTransform
from dataset.total_text import TotalText_test
from util.option import BaseOptions
from util.config import config as cfg, update_config, print_config
from torch.autograd import Variable
img_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/test/'
output_path = '/home/weijia.wu/workspace/Kaggle/Word_segementation/ronghe/two_deeplab_me/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 2
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)
model_path = ''


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def sigmoid(x):
  return 1. / (1 + np.exp(-x))
def main():
    data_Ids = [f.split('.')[0] for f in os.listdir(img_path)]
    image_datasets = TotalText_test(data_Ids=list(data_Ids), imgs_path=img_path)

    image_dataloader = data.DataLoader(image_datasets, batch_size=3, shuffle=True, num_workers=5)

    import cv2
    model = DeepLab(num_classes=3,backbone='drn',
                      output_stride=8,sync_bn=True,freeze_bn=False)
    #model = UNet(n_channels=3,n_classes=3)
    model.cuda()
    model = nn.DataParallel(model)

    model_path = os.path.join(cfg.save_dir, cfg.exp_name,'textsegementation_best_1.pth')

    # model_path = '/home/weijia.wu/workspace/Kaggle/Word_segementation/base_text_segementation_5/Demo/ouput/textsegementation_best_model.pth'
    load_model(model, model_path)

    print('Start testing TextSnake.')

    model.eval()
    for i, (image,meta) in enumerate(image_dataloader):
        image = Variable(image.cuda())
        output = model(image)
        for idx in range(image.size(0)):

            one_pred = output[idx, 0].data.cpu().numpy()
            two_pred = output[idx, 1].data.cpu().numpy()

            one_pred_ = sigmoid(one_pred)
            two_pred_ = sigmoid(two_pred)

            # one_pred_, two_pred_ = np.where(one_pred_ > 0.5, 1, 0), np.where(two_pred_ > 0.5, 1, 0)

            img_show = image[idx].permute(1, 2, 0).cpu().numpy()
            # result = one_pred_ * 60 + two_pred_ * 120

            path = os.path.join(output_path,meta['image_id'][idx]+'.npy')
            print(path)
            # np.save(path,two_pred_)
            # cv2.imwrite(path, one_pred_)
            # print('cliqueNet parameters:', sum(param.numel() for param in model.parameters()))
        break

def test():
    from sklearn.model_selection import train_test_split
    from dataset.total_text import TotalText
    from torch.utils.data import DataLoader
    from util.evaluation import evaluation
    # load data
    img_path = '/home/xjc/Desktop/yi_cut/number_segment_1/img'
    mask_path = '/home/xjc/Desktop/yi_cut/number_segment_1/label'

    data_Ids = [f.split('.')[0] for f in os.listdir(img_path)]

    # image_list = os.listdir(img_path)
    # data_Ids = [image_list[i].split('.')[0] for i in range(20000)]

    train_Ids ,valid_labels = train_test_split(data_Ids,test_size = 0.1,random_state=42)

    vilid_datasets = TotalText(data_Ids=list(valid_labels), imgs_path=img_path, masks_path=mask_path\
                              ,is_training=True, transform=BaseTransform(size=400, mean=cfg.means, std=cfg.stds))
    vilid_dataloader = DataLoader(vilid_datasets, batch_size=cfg.batch_size, shuffle=False, num_workers=15)


    model = DeepLab(num_classes=3,backbone='drn',
                      output_stride=8,sync_bn=True,freeze_bn=False)
    #model = UNet(n_channels=3,n_classes=3)
    model.cuda()
    model = nn.DataParallel(model)

    model_path = os.path.join(cfg.save_dir, cfg.exp_name, \
                              'textsegementation_best.pth'.format(cfg.backbone_name, cfg.checkepoch))

    print("TextCohesion  <==> Loading checkpoint '{}' <==> Begin".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print("TextCohesion  <==> Loading checkpoint '{}' <==> Done".format(model_path))

    model.eval()

    all_iou = 0
    with torch.no_grad():
        for i, (image, one_mask, two_mask, all_mask, meta, one_mask_weight, two_mask_weight, box_mask) in enumerate(
                vilid_dataloader):
            image = Variable(image.cuda())
            one_mask = Variable(one_mask.cuda())
            two_mask = Variable(two_mask.cuda())
            all_mask = Variable(all_mask.cuda())
            one_mask_weight = Variable(one_mask_weight.cuda())
            two_mask_weight = Variable(two_mask_weight.cuda())
            box_mask = Variable(box_mask.cuda())

            output = model(image)

            all_iou = evaluation(image, output, one_mask, two_mask, all_mask, all_iou)


    one_iou = all_iou / len(valid_labels)
    print('current_iou:  {}'.format(one_iou))


if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()
    # test()
