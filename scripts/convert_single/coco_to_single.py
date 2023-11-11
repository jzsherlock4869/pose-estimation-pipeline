import json
import cv2
import os
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO

# this script is used to transform multi-person dataset
# into single-person dataset 
src_json_path = '../annotation_conversion/mpii_json/mpii_cocoformat_train.json'
src_img_dir = '/Users/jzsherlock/datasets/MPII/images'
dst_json_path = './mpii_single/mpii_single_cocoformat_train.json'
dst_img_dir = '/Users/jzsherlock/datasets/MPII/images_single'
imgsize = 640
bbox_pad = 30

# ==== start single-pose conversion ==== #

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(os.path.dirname(dst_json_path), exist_ok=True)

single_coco = {'images': [], 'categories': [], 'annotations': []}

# add category
category = {
    "supercategory": "person",
    "id": 1,  # to be same as COCO, not using 0
    "name": "person",
    "skeleton":
        [[0,1],
            [1,2],
            [2,6],
            [7,12],
            [12,11],
            [11,10],
            [5,4],
            [4,3],
            [3,6],
            [7,13],
            [13,14],
            [14,15],
            [6,7],
            [7,8],
            [8,9]],
    "keypoints": 
        [
            "r_ankle", "r_knee","r_hip", 
            "l_hip", "l_knee", "l_ankle",
            "pelvis", "throax","upper_neck", "head_top",
            "r_wrist", "r_elbow", "r_shoulder",
            "l_shoulder", "l_elbow", "l_wrist"]
}

single_coco['categories'] = [category]
print('add category done')


coco = COCO(annotation_file=src_json_path)
# get all image index info
ids = list(sorted(coco.imgs.keys()))
print("number of images: {}".format(len(ids)))
skeleton = coco.cats[1]['skeleton']
np.random.seed(42)
colormap =  np.random.randint(0, 255, [len(skeleton), 3], np.uint8)

# get all coco class labels
coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])
print('class: ', coco_classes)

cnt = 0
pbar = tqdm(ids, total=len(ids))
for imid in pbar:
    im_info = coco.loadImgs(imid)[0]
    imname = im_info['file_name']
    impath = os.path.join(src_img_dir, imname)
    HH, WW = im_info['height'], im_info['width']
    # print(f'[image info] file path: {impath}, HxW: {HH} x {WW}')

    img = cv2.imread(impath)
    ann_ids = coco.getAnnIds(imgIds=imid)
    annos = coco.loadAnns(ann_ids)
    for anno_idx, anno in enumerate(annos):
        x, y, w_b, h_b = anno['bbox']
        kpts = anno['keypoints']
        num_kpts = anno['num_keypoints']
        x1, x2 = max(0, x - bbox_pad), min(WW, x + w_b + bbox_pad)
        y1, y2 = max(0, y - bbox_pad), min(HH, y + h_b + bbox_pad)
        w, h = x2 - x1, y2 - y1
        ratio = imgsize / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        new_w_b, new_h_b = int(w_b * ratio), int(h_b * ratio)
        
        img_sin = cv2.resize(img[y1:y2, x1:x2], 
                             (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_pad = np.zeros((imgsize, imgsize, 3), dtype=np.uint8)
        start_h = imgsize // 2 - new_h // 2
        start_w = imgsize // 2 - new_w // 2
        img_pad[start_h: start_h + new_h, start_w: start_w + new_w, :] = img_sin

        kpts = np.array(kpts).reshape(-1, 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for kpt_idx in range(len(kpts)):
            if kpts[kpt_idx, 2] == 1:
                kx, ky = kpts[kpt_idx, :2]
                new_kx = int((kx - cx) * ratio) + imgsize // 2
                new_ky = int((ky - cy) * ratio) + imgsize // 2
                kpts[kpt_idx, 0] = new_kx
                kpts[kpt_idx, 1] = new_ky
                # cv2.circle(img_pad, (new_kx, new_ky), 10, (0, 0, 255), -1)

        # ====== show skeleton to validate correctness ===== #
        # for ln_idx, (src, dst) in enumerate(skeleton):
        #     if kpts[src, 2] == 1 and kpts[dst, 2] == 1:
        #         kx_s, ky_s = kpts[src, :2]
        #         kx_d, ky_d = kpts[dst, :2]
        #         sk_color = colormap[ln_idx, :].tolist()
        #         cv2.line(img_pad, (kx_s, ky_s), (kx_d, ky_d), sk_color, 3)

        filename = imname.split('.')[0] + f'_{anno_idx}.' + imname.split('.')[1]
        img_dict = {'id': cnt, 'file_name': filename, 'width': imgsize, 'height': imgsize}
        single_coco['images'].append(img_dict)
        cv2.imwrite(os.path.join(dst_img_dir, filename), img_pad)

        if x - bbox_pad < 0:
            delta_x = int((2 * bbox_pad - x) * ratio)
        else:
            delta_x = int(bbox_pad * ratio)
        if y - bbox_pad < 0:
            delta_y = int((2 * bbox_pad - y) * ratio)
        else:
            delta_y = int(bbox_pad * ratio)
        new_bbox = [start_w + delta_x, start_h + delta_y,
                    new_w_b, new_h_b]

        per_dict = {'id': cnt,
                    'image_id': cnt,
                    'category_id': 1,
                    'area': int(new_bbox[2] * new_bbox[3]),
                    'bbox': new_bbox,
                    'iscrowd': 0,
                    'keypoints': kpts.reshape(-1).tolist(),
                    'num_keypoints': int(np.sum(kpts[:, 2]))}

        single_coco['annotations'].append(per_dict)

        cnt += 1
        
        # cv2.rectangle(img_pad, (new_bbox[0], new_bbox[1]),
        #             (new_bbox[0] + new_bbox[2], new_bbox[1] + new_bbox[3]),
        #             color=(0,255,255), thickness=2)

        # cv2.namedWindow('demo')
        # cv2.imshow('demo', img_pad)
        # cv2.waitKey(0)

print('add annotations and images done')

with open(dst_json_path, 'w') as fcoco:
    json.dump(single_coco, fcoco, indent=4, cls=NpEncoder)

