import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from data_utils import gen_center_heatmap, gen_kpt_heatmap, \
                    gen_kpt_regression, gen_offset_regression

import cv2
from glob import glob


class CocoSinglePoseDataset(Dataset):
    """
        COCO dataset for single person pose estimation (keypoint detection)
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot_img = opt_dataset['dataroot_img']
        self.dataroot_json = opt_dataset['dataroot_json']
        self.phase = opt_dataset['phase']

        augment_opt = opt_dataset['augment']
        insize, outsize = augment_opt['size'], opt_dataset['out_size']
        self.in_size = (insize, insize)
        self.out_size = (outsize, outsize) # output size of feature map
        self.sigma = opt_dataset['sigma'] # sigma of blur for keypoint in heatmap
        augment_type = augment_opt.pop('augment_type')
        if self.phase == 'train':
            self.augment = importlib.\
                import_module(f'data_augment.{augment_type}').train_augment(**augment_opt)
        elif self.phase == 'valid':
            self.augment = importlib.\
                import_module(f'data_augment.{augment_type}').val_augment(**augment_opt)

        self.coco = COCO(annotation_file=self.dataroot_json)
        # get all image index info
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        imid = self.ids[idx]
        im_info = self.coco.loadImgs(imid)[0]
        imname = im_info['file_name']
        impath = os.path.join(self.dataroot_img, imname)
        cur_img = cv2.imread(impath)
        ori_w, ori_h = im_info['width'], im_info['height']
        ann_ids = self.coco.getAnnIds(imgIds=imid)
        # only for single-person dataset 
        anno = self.coco.loadAnns(ann_ids)[0]
        bbox = anno['bbox']
        kpts = anno['keypoints']
        ratio_w = ori_w / self.in_size[0]
        ratio_h = ori_h / self.in_size[1]
        bbox = [int(bbox[0] * ratio_w), int(bbox[1] * ratio_h),
                 int(bbox[2] * ratio_w), int(bbox[3] * ratio_h)]
        kpts = np.array(kpts).reshape(-1, 3)
        kpts[:, 0] = (kpts[:, 0] * ratio_w).astype(np.int64)
        kpts[:, 1] = (kpts[:, 1] * ratio_h).astype(np.int64)
        hm_center = gen_center_heatmap(bbox, self.in_size, self.out_size, self.sigma)
        hm_keypoint = gen_kpt_heatmap(bbox, kpts, self.in_size, self.out_size, self.sigma)
        map_reg_x, map_reg_y = gen_kpt_regression(bbox, kpts, self.in_size, self.out_size)
        map_offset_x, map_offset_y = gen_offset_regression(bbox, kpts, self.in_size, self.out_size)
        img_aug = self.augment(image=cur_img)['image']
        output_dict = {
            "img": img_aug,
            "hm_center": hm_center,
            "hm_keypoint": hm_keypoint,
            "map_reg_x": map_reg_x,
            "map_reg_y": map_reg_y,
            "map_offset_x": map_offset_x,
            "map_offset_y": map_offset_y,
            "imname": imname,
            "ori_size": (ori_h, ori_w)
        }
        return output_dict

    def __len__(self):
        return len(self.ids)


def CocoSinglePoseDataloader(opt_dataloader):
    phase = opt_dataloader['phase']
    if phase == 'train':
        batch_size = opt_dataloader['batch_size']
        num_workers = opt_dataloader['num_workers']
        shuffle = True
    elif phase == 'valid':
        batch_size = 1
        num_workers = 0
        shuffle = False
    folder_dataset = CocoSinglePoseDataset(opt_dataloader)
    dataloader = DataLoader(folder_dataset, batch_size=batch_size, pin_memory=True, \
                            drop_last=True, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    opt_dataset = {
        'dataroot_img': '/Users/jzsherlock/datasets/MPII/images_single',
        'dataroot_json': '../scripts/convert_single/mpii_single/mpii_single_cocoformat_train.json',
        'phase': 'train',
        'sigma': 3,
        'augment':
        {
            'augment_type': 'simple_aug',
            'size': 640
        },
        'out_size': 48,
        'batch_size': 2,
        'num_workers': 4
    }
    dataloader = CocoSinglePoseDataloader(opt_dataset)
    for idx, batch in enumerate(dataloader):
        if idx > 3:
            break
        print(idx)
        print(batch['img'].shape, batch['imname'], batch['ori_size'])
        print(batch['hm_center'].shape, batch['hm_keypoint'].shape)
        print(batch['map_reg_x'].shape, batch['map_reg_y'].shape)
        print(batch['map_offset_x'].shape, batch['map_offset_y'].shape)

