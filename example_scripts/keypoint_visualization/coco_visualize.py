import os
import cv2
import numpy as np
from pycocotools.coco import COCO

json_path = '../annotation_conversion/mpii_json/mpii_cocoformat_train.json'
img_dir = '/Users/jzsherlock/datasets/MPII/images'
coco = COCO(annotation_file=json_path)

# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))
skeleton = coco.cats[1]['skeleton']
np.random.seed(42)
colormap =  np.random.randint(0, 255, [len(skeleton), 3], np.uint8)

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
print('class: ', coco_classes)

for imid in ids[1:30]:
    im_info = coco.loadImgs(imid)[0]
    impath = os.path.join(img_dir, im_info['file_name'])
    HH, WW = im_info['height'], im_info['width']
    print(f'[image info] file path: {impath}, HxW: {HH} x {WW}')

    img = cv2.imread(impath)
    ann_ids = coco.getAnnIds(imgIds=imid)
    annos = coco.loadAnns(ann_ids)
    for anno in annos:
        x, y, w, h = anno['bbox']
        kpts = anno['keypoints']
        num_kpts = anno['num_keypoints']
        x1, x2, y1, y2 = x, x + w, y, y + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,255,255), thickness=2)
        kpts = np.array(kpts).reshape(-1, 3)
        for kpt_idx in range(len(kpts)):
            if kpts[kpt_idx, 2] == 1:
                kx, ky = kpts[kpt_idx, :2]
                cv2.circle(img, (kx, ky), 10, (0, 0, 255), -1)
        for ln_idx, (src, dst) in enumerate(skeleton):
            if kpts[src, 2] == 1 and kpts[dst, 2] == 1:
                kx_s, ky_s = kpts[src, :2]
                kx_d, ky_d = kpts[dst, :2]
                sk_color = colormap[ln_idx, :].tolist()
                cv2.line(img, (kx_s, ky_s), (kx_d, ky_d), sk_color, 3)
    
    cv2.namedWindow('demo')
    cv2.imshow('demo', img)
    cv2.waitKey(0)

