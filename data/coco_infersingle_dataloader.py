import os, sys
import importlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO

import cv2
from glob import glob


class CocoInferSingleDataset(Dataset):
    """
    Images for inference for single-pose estimation
    """
    def __init__(self, opt_dataset) -> None:
        super().__init__()
        # parse used arguments, explicit parsing is easier for debug
        self.dataroot_img = opt_dataset['dataroot_img']
        self.dataroot_json = opt_dataset['dataroot_json']
        self.phase = opt_dataset['phase']

        augment_opt = opt_dataset['augment']
        # insize = augment_opt['size']
        # self.in_size = (insize, insize)
        augment_type = augment_opt.pop('augment_type')
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
        img_aug = self.augment(image=cur_img)['image']
        # simple dataset cannot cover mixup/contrast etc. which need 2 or more images to return
        output_dict = {
            "img": img_aug,
            "img_path": impath,
            "ori_size_wh": (ori_w, ori_h)
        }
        return output_dict

    def __len__(self):
        return len(self.ids)


def CocoInferSingleDataloader(opt_dataloader):
    folder_dataset = CocoInferSingleDataset(opt_dataloader)
    dataloader = DataLoader(folder_dataset, batch_size=1, pin_memory=True, \
                            drop_last=True, shuffle=False, num_workers=0)
    return dataloader


if __name__ == "__main__":
    opt_dataset = {
        'dataroot_img': '/Users/jzsherlock/datasets/MPII/images_single',
        'dataroot_json': '../scripts/convert_single/mpii_single/mpii_single_cocoformat_train.json',
        'phase': 'test',
        'augment':
        {
            'augment_type': 'simple_aug',
            'size': 640
        },
        'batch_size': 2,
        'num_workers': 4
    }
    dataloader = CocoInferSingleDataloader(opt_dataset)
    for idx, batch in enumerate(dataloader):
        if idx > 3:
            break
        print(idx)
        print(batch['img'].shape, batch['img_path'], batch['ori_size_wh'])
