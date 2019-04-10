import copy
import cv2
import os
import torch.utils.data as data
import scipy.io as io
import numpy as np
import torch
from scipy import ndimage as ndi
from util.augmentation import BaseTransform
from dataset.data_util import pil_load_img
from skimage import morphology, draw
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from dataset.data_util import min_area_rect, rect_to_xys
from skimage.draw import polygon as drawpoly


def Normalize(image):
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    image = image.astype(np.float32)
    image /= 255.0
    image -= mean
    image /= std
    return image


class TotalText_test(data.Dataset):

    def __init__(self, data_Ids, imgs_path):
        super().__init__()
        self.data_Ids = data_Ids
        self.imgs_path = imgs_path

    def __getitem__(self, item):
        img_name = self.data_Ids[item]
        image = pil_load_img(path=os.path.join(self.imgs_path, img_name + '.jpg'))

        H, W, _ = image.shape
        image = Normalize(image)
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()

        meta = {
            'image_id': img_name,
            'Height': H,
            'Width': W
        }

        return image, meta

    def __len__(self):
        return len(self.data_Ids)


class TotalText(data.Dataset):

    def __init__(self, data_Ids, imgs_path, masks_path, is_training=True, transform=None):
        super().__init__()
        self.data_Ids = data_Ids
        self.imgs_path = imgs_path
        self.annotation_root = masks_path
        self.is_training = is_training
        self.transform = transform

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0])
        mask[rr, cc] = value

    def __getitem__(self, item):
        # img_name = self.data_Ids[item]
        img_name = 'img_254'
        image = pil_load_img(path=os.path.join(self.imgs_path, img_name + '.jpg'))
        # label_name = img_name.split('_')[1]
        label_name = '254'
        label = pil_load_img(path=os.path.join(self.annotation_root, 'label_' + label_name + '.png'))

        H, W, _ = image.shape
        if self.transform:
            image, label = self.transform(image, label)

        # create labels
        # one_mask = np.zeros(label.shape,np.uint8)
        # two_mask = np.zeros(label.shape,np.uint8)
        mask = np.ones(label.shape, np.uint8)
        box_mask = np.zeros(label.shape, np.uint8)
        # rectangular_box = np.zeros(label.shape[:2], np.uint8)

        # get two label with overlap text
        one_mask = ((np.array(label) > 20) * (np.array(label) < 100)).astype(np.uint8) * 1 + \
                   ((np.array(label) > 120) * (np.array(label) < 220)).astype(np.uint8) * 1
        two_mask = ((np.array(label) > 80) * (np.array(label) < 160)).astype(np.uint8) * 1 + \
                   ((np.array(label) > 120) * (np.array(label) < 220)).astype(np.uint8) * 1

        all_mask = one_mask + two_mask

        poinrs = np.where(all_mask)
        y_min, y_max, x_min, x_max = poinrs[0].min(), poinrs[0].max(), poinrs[1].min(), poinrs[1].max()
        x_ = [x_min, x_max, x_max, x_min]
        y_ = [y_min, y_min, y_max, y_max]
        pointss = np.stack([x_, y_]).T.astype(np.int32)
        self.fill_polygon(box_mask, pointss, value=1)

        # kernel = np.ones((4, 4), np.uint8)

        # overlap_area = one_mask * two_mask - cv2.erode(one_mask * two_mask, kernel)

        # 实施骨架算法

        # one_mask_skleton = morphology.skeletonize(one_mask).astype(np.uint8) * 1
        # two_mask_skleton = morphology.skeletonize(two_mask).astype(np.uint8) * 1
        # one_mask_skleton = cv2.erode(one_mask, kernel)  # 膨胀图像
        # two_mask_skleton = cv2.erode(two_mask, kernel)  # 膨胀图像
        #
        # one_mask_weight_1 = one_mask - one_mask_skleton
        # two_mask_weight_2 = two_mask - two_mask_skleton
        #分水岭算法
        one_mask_weight = ndi.distance_transform_edt(one_mask) #距离变换
        two_mask_weight = ndi.distance_transform_edt(two_mask) #距离变换
        one_mask_weight_1 = -(one_mask_weight - one_mask_weight.max() * one_mask)/one_mask_weight.max() * 3 + 1
        two_mask_weight_2 = -(two_mask_weight - two_mask_weight.max() * two_mask)/two_mask_weight.max() * 3+ 1


        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        one_mask = torch.from_numpy(one_mask).float()
        two_mask = torch.from_numpy(two_mask).float()
        all_mask = torch.from_numpy(all_mask).float()
        one_mask_weight = torch.from_numpy(one_mask_weight_1).float()
        two_mask_weight = torch.from_numpy(two_mask_weight_2).float()


        meta = {
            'image_id': img_name + '.jpg',
            'image_path': os.path.join(self.imgs_path, img_name + '.jpg'),
            'Height': H,
            'Width': W
        }

        return image, one_mask, two_mask, all_mask, meta, one_mask_weight, two_mask_weight, box_mask

    def __len__(self):
        return len(self.data_Ids)


def points_to_contours(points):
    return np.asarray([points_to_contour(points)])


def points_to_contour(points):
    contours = [[list(p)] for p in points]
    return np.asarray(contours, dtype=np.int32)


if __name__ == '__main__':
    from util.option import BaseOptions
    from util.augmentation import BaseTransform, Augmentation
    from util.config import config as cfg, update_config, print_config

    option = BaseOptions()
    args = option.initialize()
    update_config(cfg, args)

    from torch.utils.data import DataLoader

    img_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/img'
    mask_path = '/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/label'
    data_Ids = [f.split('.')[0] for f in os.listdir(img_path)]

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    image_datasets = TotalText(data_Ids=list(data_Ids), imgs_path=img_path, masks_path=mask_path \
                               , is_training=True, transform=BaseTransform(size=400, mean=means, std=stds))
    image_dataloader = DataLoader(image_datasets, batch_size=1, shuffle=True, num_workers=5)
    image, one_mask, two_mask, all_mask, meta, one_mask_weight, two_mask_weight, box_mask = next(iter(image_dataloader))
    print('image:', image.shape)
    print(meta)
    print('train_mask:', one_mask.shape)
    print('tr_mask:', two_mask.shape)
    print('tcl_mask:', all_mask.shape)

    image = image[0].numpy() * 255
    # label =label[0].numpy()
    one_mask = one_mask[0].numpy() * 255
    two_mask = two_mask[0].numpy() * 255
    all_mask = all_mask[0].numpy() * 255
    one_mask_weight = one_mask_weight[0].numpy()
    two_mask_weight = two_mask_weight[0].numpy()

    np.savetxt(cfg.vis_dir + 'one_mask_weight.txt',
               one_mask_weight, fmt='%0.2f')
    np.savetxt(cfg.vis_dir + 'two_mask_weight.txt',
               two_mask_weight, fmt='%0.2f')

    poinrs = np.where(all_mask)
    print(poinrs[0].min(), poinrs[0].max(), poinrs[1].min(), poinrs[1].max())



    image = np.asarray(image.transpose(1, 2, 0))

    path1 = os.path.join(cfg.vis_dir, 'image.jpg')
    path2 = os.path.join(cfg.vis_dir, 'label.jpg')
    path3 = os.path.join(cfg.vis_dir, 'one_mask.jpg')
    path4 = os.path.join(cfg.vis_dir, 'two_mask.jpg')
    path5 = os.path.join(cfg.vis_dir, 'all_mask.jpg')
    path6 = os.path.join(cfg.vis_dir, 'one_mask_skleton.jpg')
    path7 = os.path.join(cfg.vis_dir, 'two_mask_skleton.jpg')
    path8 = os.path.join(cfg.vis_dir, 'rectangular_box.jpg')
    path9 = os.path.join(cfg.vis_dir, 'two_mask_weight.jpg')
    path10 = os.path.join(cfg.vis_dir, 'one_mask_weight.jpg')

    print(image.shape)

    cv2.imwrite(path1, image)

    cv2.imwrite(path3, one_mask)
    cv2.imwrite(path4, two_mask)
    cv2.imwrite(path5, all_mask)
    cv2.imwrite(path9, two_mask_weight)
    cv2.imwrite(path10, one_mask_weight)
