import os
import time
import torch
from torch.optim import lr_scheduler
from dataset.total_text import TotalText
from network.vgg_unet import UnetVgg11
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from network.loss import TextLoss
from util.augmentation import BaseTransform, Augmentation
from torch.autograd import Variable
from util.misc import AverageMeter
from util.config import config as cfg, update_config, print_config
from sklearn.model_selection import train_test_split
from util.evaluation import evaluation
from util.option import BaseOptions

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# training

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = cfg.lr * (0.8 ** (epoch // 5))
    if epoch % 3 == 0 and epoch > 80:
        lr = cfg.lr * (0.9 ** ((epoch - 80) // 8))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('current lr : ', param_group['lr'])


def train(model, train_loader, criterion, scheduler, optimizer, epoch):
    losses = AverageMeter()
    start = time.time()
    end = time.time()

    model.train()

    adjust_learning_rate(optimizer, epoch)
    for i, (image, one_mask, two_mask, all_mask, meta, one_mask_weight, two_mask_weight, box_mask) in enumerate(
            train_loader):
        image = Variable(image.cuda())
        one_mask = Variable(one_mask.cuda())
        two_mask = Variable(two_mask.cuda())
        all_mask = Variable(all_mask.cuda())
        one_mask_weight = Variable(one_mask_weight.cuda())
        two_mask_weight = Variable(two_mask_weight.cuda())
        box_mask = Variable(box_mask.cuda())

        output = model(image)

        bce_loss, dice_loss, loss_box = \
            criterion(output, one_mask, two_mask, all_mask, one_mask_weight, two_mask_weight, box_mask)
        loss = bce_loss + dice_loss

        # print("2")
        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())

        if i % cfg.display_freq == 0:
            print(
                'Epoch: [ {} ][ {:03d} / {:03d} ] - Loss: {:.6f} - bce_loss: {:.6f} - dice_loss: {:.6f}'.format(
                    epoch, i, len(train_loader), loss.item(), bce_loss.item(), dice_loss.item())
            )

    print('Training Loss: {}'.format(losses.avg))


def validation(model, vilid_dataloader, criterion, scheduler, optimizer, epoch, IOU, valid_labels):
    model.eval()
    losses = AverageMeter()

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

            bce_loss, dice_loss, loss_box = \
                criterion(output, one_mask, two_mask, all_mask, one_mask_weight, two_mask_weight, box_mask)
            loss = bce_loss + dice_loss + loss_box * 0.01

            losses.update(loss.item())

            if i % cfg.display_freq == 0:
                print(
                    'Validation:  Epoch - [ {} ][ {:03d} / {:03d} ] - Loss: {:.6f}  - bce_loss: {:.6f} - dice_loss: {:.6f}'.format(
                        epoch, i, len(vilid_dataloader), loss.item(), bce_loss.item(), dice_loss.item())
                )

            if epoch % cfg.evaluation_freq == 0:
                all_iou = evaluation(image, output, one_mask, two_mask, all_mask, all_iou)

    one_iou = 0

    print('Validation Loss: {}'.format(losses.avg))
    if epoch % cfg.evaluation_freq == 0:
        one_iou = all_iou / len(valid_labels)

        if one_iou > IOU:
            IOU = one_iou
            save_model(model, epoch, scheduler.get_lr(), optimizer)

        print('current_iou:  {}'.format(one_iou))
        print('Max_IOU:  {}'.format(IOU))
    return IOU


def save_model(model, epoch, lr, optimizer):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'textsegementation_best.pth')
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state_dict, save_path)


def main():
    # load data
    img_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/img'
    mask_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/label'

    data_Ids = [f.split('.')[0] for f in os.listdir(img_path)]

    # image_list = os.listdir(img_path)
    # data_Ids = [image_list[i].split('.')[0] for i in range(20000)]

    train_Ids ,valid_labels = train_test_split(data_Ids,test_size = 0.2,random_state=42)




    print('train_Ids',len(train_Ids))
    print('valid_Ids',len(valid_labels))
    image_datasets = TotalText(data_Ids=list(train_Ids), imgs_path=img_path, masks_path=mask_path\
                              ,is_training=True, transform=Augmentation(size=400, mean=cfg.means, std=cfg.stds))
    image_dataloader = DataLoader(image_datasets, batch_size=cfg.batch_size, shuffle=True, num_workers=15)

    vilid_datasets = TotalText(data_Ids=list(valid_labels), imgs_path=img_path, masks_path=mask_path\
                              ,is_training=True, transform=BaseTransform(size=400, mean=cfg.means, std=cfg.stds))
    vilid_dataloader = DataLoader(vilid_datasets, batch_size=cfg.batch_size, shuffle=False, num_workers=15)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UnetVgg11(n_classes=3)
    model.cuda()
    model = nn.DataParallel(model)

    # loss
    criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)

    model_path = os.path.join(cfg.save_dir, cfg.exp_name,'textsegementation_best.pth')
    # init or resume
    if cfg.resume and os.path.isfile(model_path):
        print("TextCohesion  <==> Loading checkpoint '{}' <==> Begin".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("TextCohesion  <==> Loading checkpoint '{}' <==> Done".format(model_path))
    else:
        start_epoch = 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
        print('current lr : ', param_group['lr'])
        
    print('Start training.')

    IOU = 0.965
    for epoch in range(start_epoch, cfg.max_epoch):
        train(model, image_dataloader, criterion, scheduler, optimizer, epoch)
        IOU = validation(model,vilid_dataloader,criterion,scheduler,optimizer,epoch,IOU,valid_labels)
    print("final IOU: ", IOU)
    print('End.')

if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()

