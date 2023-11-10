import os
import os.path as osp
import numpy as np
import itertools
from copy import deepcopy
import json
from tqdm import tqdm
import cv2

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def convert_MPIIjson_cocoformat(json_path, image_dir, coco_path, phase='train'):

    with open(json_path, 'r') as fjson:
        mpii_json = json.load(fjson)
    coco = {'images': [], 'categories': [], 'annotations': []}

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

    coco['categories'] = [category]
    print('add category done')

    ############# filter train or test annotations #############
    if phase == 'train':
        is_phase_ls = mpii_json['img_train']  # [1, 1, 1, 1, 0, 0, ...] 
    else:
        assert phase == 'test'
        is_phase_ls = [1 - i for i in mpii_json['img_train']]
    
    phase_idx_ls = np.where(np.array(is_phase_ls) == 1)[0].tolist()  # [ 4, 5, 6, ..., 24982, 24983, 24984]
    annolist = [mpii_json['annolist'][idx] for idx in phase_idx_ls]
    total_num_sample = len(phase_idx_ls)

    ############## add image samples #############
    total_per_idx = 0
    pbar = tqdm(range(total_num_sample), total=total_num_sample)
    # pbar = tqdm(range(17000, 18079), total=1079)  # for debug
    for imid in pbar:
        sample = annolist[imid]
        filename = sample['image']['name']
        impath = osp.join(image_dir, filename)
        img = cv2.imread(impath)
        if img is None:
            print(f'[warning] missing image {filename} in given folder {image_dir}')
            continue
        height, width, _ = img.shape
        # TODO: check if the imid can overlap in train/test or as split
        img_dict = {'id': imid, 'file_name': filename, 'width': width, 'height': height}
        coco['images'].append(img_dict)

        ############### add annotations ###############
        if phase == 'train':
            annorect_ls = sample['annorect']
            if isinstance(annorect_ls, dict):
                annorect_ls = [annorect_ls]
            assert isinstance(annorect_ls, list)
            num_person = len(annorect_ls)
            for per_idx in range(num_person):
                if 'annopoints' not in annorect_ls[per_idx]:
                    continue
                if len(annorect_ls[per_idx]['annopoints']) == 0:
                    continue
                cur_joint_ls = annorect_ls[per_idx]['annopoints']['point']
                bbox = np.zeros((4), dtype='int') # xmin, ymin, w, h
                kps = np.zeros((16, 3), dtype='int') # xcoord, ycoord, vis
                if not isinstance(cur_joint_ls, list):
                    cur_joint_ls = [cur_joint_ls]
                for cur_joint in cur_joint_ls:
                    x, y, jid = int(cur_joint['x']), int(cur_joint['y']), int(cur_joint['id'])
                    kps[jid][0], kps[jid][1], kps[jid][2] = x, y, 1
                #bbox extract from annotated kps
                anno_kps = kps[kps[:,2] == 1, :].reshape(-1,3)
                xmin = np.min(anno_kps[:,0])
                ymin = np.min(anno_kps[:,1])
                xmax = np.max(anno_kps[:,0])
                ymax = np.max(anno_kps[:,1])
                width = xmax - xmin - 1
                height = ymax - ymin - 1
                # corrupted bounding box
                if width <= 0 or height <= 0:
                    continue
                else:
                    # ext percentage extend
                    # TODO: consider border effect
                    ext = 0.0
                    bbox[0] = (xmin + xmax) / 2. - width / 2 * (1 + ext)
                    bbox[1] = (ymin + ymax) / 2. - height / 2 * (1 + ext)
                    bbox[2] = width * (1 + ext)
                    bbox[3] = height * (1 + ext)

                per_dict = {'id': total_per_idx,
                            'image_id': imid,
                            'category_id': 1,
                            'area': int(bbox[2] * bbox[3]),
                            'bbox': bbox.tolist(),
                            'iscrowd': 0,
                            'keypoints': kps.reshape(-1).tolist(),
                            'num_keypoints': int(np.sum(kps[:, 2]))}

                coco['annotations'].append(per_dict)
                total_per_idx += 1

    print('add annotations and images done')

    with open(coco_path, 'w') as fcoco:
        json.dump(coco, fcoco, indent=4, cls=NpEncoder)


if __name__ == "__main__":
    
    json_path = './mpii_json/mpii_anno.json'
    # image_dir = '/data/public_dataset/MPII/images'
    image_dir = '/Users/jzsherlock/datasets/MPII/images'
    coco_path = './mpii_json/mpii_cocoformat_train.json'

    convert_MPIIjson_cocoformat(json_path=json_path, image_dir=image_dir, coco_path=coco_path, phase='train')