import os
import  numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from network.textnet import TextNet
from util.augmentation import BaseTransform
from dataset.total_text import Testloader,TotalText
from util.option import BaseOptions
from util.config import config as cfg, update_config, print_config

img_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/img/'
mask_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/label'
output_path = '/home/weijia.wu/workspace/Kaggle/Word_segementation/text_segementation/Demo/output/'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 2
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)
model_path = ''


def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])


def main():
    data_Ids = [f.split('.')[0] for f in os.listdir(img_path)]
    image_datasets = TotalText(data_Ids=list(data_Ids), imgs_path=img_path, masks_path=mask_path\
                              ,is_training=True, transform=BaseTransform(size=512, mean=cfg.means, std=cfg.stds))

    # image_datasets = Testloader(data_Ids=list(data_Ids), imgs_path=img_path \
    #                             , is_training=True, transform=BaseTransform(size=512, mean=means, std=stds))
    image_dataloader = data.DataLoader(image_datasets, batch_size=cfg.batch_size, shuffle=True, num_workers=5)

    from torch.autograd import Variable
    import cv2
    model = TextNet()
    model.cuda()
    model = nn.DataParallel(model)
    # model = nn.DataParallel(model)

    model_path = '/home/weijia.wu/workspace/Kaggle/Word_segementation/text_segementation/save_model/test_1/textsegementation_VGG16_15.pth'
    load_model(model, model_path)

    print('Start testing TextSnake.')

    image, label, one_mask, two_mask, all_mask, meta = next(iter(image_dataloader))

    output = model(image)
    for idx in range(image.size(0)):
        one_pred = output[idx, 0].data.cpu().numpy()
        two_pred = output[idx, 1].data.cpu().numpy()
        all_pred = output[idx, 2].data.cpu().numpy()

        label_ = label[idx].numpy()
        one_mask_ = one_mask[idx].numpy() * 255
        two_mask_ = two_mask[idx].numpy() * 255
        all_mask_ = all_mask[idx].numpy() * 255

        one_pred = (np.array(one_pred) > cfg.confi_thresh).astype(np.uint8) * 1
        two_pred = (np.array(two_pred) > cfg.confi_thresh).astype(np.uint8) * 1
        all_pred = (np.array(all_pred) > cfg.confi_thresh).astype(np.uint8) * 1

        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        #     img_show = ((img_show * stds + means) * 255).astype(np.uint8)

        one_pred = cv2.cvtColor(one_pred * 255, cv2.COLOR_GRAY2BGR)
        two_pred = cv2.cvtColor(two_pred * 255, cv2.COLOR_GRAY2BGR)
        all_pred = cv2.cvtColor(all_pred * 255, cv2.COLOR_GRAY2BGR)

        tr_show = np.concatenate([img_show, all_pred], axis=1)
        tc_show = np.concatenate([one_pred, two_pred], axis=1)
        show = np.concatenate([tr_show, tc_show], axis=0)
        show = cv2.resize(show, (512, 512))

        path = os.path.join(output_path,'result.jpg')
        cv2.imwrite(path, show)

        tr_show = np.concatenate([label_, all_mask_], axis=1)
        tc_show = np.concatenate([one_mask_, two_mask_], axis=1)
        show = np.concatenate([tr_show, tc_show], axis=0)
        show = cv2.resize(show, (512, 512))

        path = os.path.join(output_path,'result1.jpg')
        cv2.imwrite(path, show)
if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()