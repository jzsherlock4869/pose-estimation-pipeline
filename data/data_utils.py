import cv2
import numpy as np

def point_to_heatmap(cx, cy, width, height, sigma=3):
    hm = np.zeros(shape=(height, width), dtype=np.float64)
    left, right = max(0, cx - 2 * sigma), min(width, cx + 2 * sigma)
    top, bottom = max(0, cy - 2 * sigma), min(height, cy + 2 * sigma)
    for x in range(left, right):
        for y in range(top, bottom):
            dist = (x - cx) ** 2 + (y - cy) ** 2
            val = np.exp(-dist / (sigma ** 2))
            hm[y, x] += val
    hm[hm > 1] = 1
    return hm

def gen_center_heatmap(bbox, input_size, output_size, sigma):
    # bbox is according to input
    # size: (w, h)
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    cx, cy = int(cx * ratio_w), int(cy * ratio_h)
    hm = point_to_heatmap(cx, cy, output_size[0], output_size[1], sigma)
    hm = np.expand_dims(hm, axis=0)
    return hm

def gen_kpt_heatmap(bbox, kpts, input_size, output_size, sigma):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    cx, cy = int(cx * ratio_w), int(cy * ratio_h)
    kpts_fm = kpts.copy()
    kpts_fm[:, 0] = (kpts_fm[:, 0] * ratio_w).astype(np.int64)
    kpts_fm[:, 1] = (kpts_fm[:, 1] * ratio_h).astype(np.int64)
    hm = np.zeros(shape=(output_size[1], output_size[0]), dtype=np.float64)
    for kpt_idx in range(len(kpts_fm)):
        if kpts_fm[kpt_idx, 2] == 1:
            kx, ky = kpts_fm[kpt_idx, :2]
            hm += point_to_heatmap(kx, ky, 
                                   output_size[0], output_size[1], sigma)
    hm = np.expand_dims(hm, axis=0)
    return hm

def gen_kpt_regression(bbox, kpts, input_size, output_size):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    cx, cy = int(cx * ratio_w), int(cy * ratio_h)
    kpts_rel = kpts.copy()
    kpts_rel[:, 0] = (kpts_rel[:, 0] * ratio_w).astype(np.int64) - cx
    kpts_rel[:, 1] = (kpts_rel[:, 1] * ratio_h).astype(np.int64) - cy
    num_kpts = len(kpts_rel)
    hm_x = np.zeros(shape=(num_kpts, output_size[1], output_size[0]), dtype=np.float64)
    hm_y = np.zeros(shape=(num_kpts, output_size[1], output_size[0]), dtype=np.float64)
    for kpt_idx in range(len(kpts_rel)):
        if kpts_rel[kpt_idx, 2] == 1:
            kx, ky = kpts_rel[kpt_idx, :2]
            hm_x[kpt_idx, cy, cx] = kx
            hm_y[kpt_idx, cy, cx] = ky
    return hm_x, hm_y

def gen_offset_regression(bbox, kpts, input_size, output_size):
    x, y, w, h = bbox
    cx, cy = x + w // 2, y + h // 2
    ratio_w = output_size[0] / input_size[0]
    ratio_h = output_size[1] / input_size[1]
    cx, cy = int(cx * ratio_w), int(cy * ratio_h)
    kpts_fm = kpts.copy().astype(np.float32)
    kpts_fm[:, 0] = kpts[:, 0] * ratio_w
    kpts_fm[:, 1] = kpts[:, 1] * ratio_h
    offsets = kpts_fm - kpts_fm.astype(np.int64)
    num_kpts = len(kpts_fm)
    hm_x = np.zeros(shape=(num_kpts, output_size[1], output_size[0]), dtype=np.float32)
    hm_y = np.zeros(shape=(num_kpts, output_size[1], output_size[0]), dtype=np.float32)
    for kpt_idx in range(len(kpts_fm)):
        if kpts_fm[kpt_idx, 2] == 1:
            kx, ky = kpts_fm[kpt_idx, :2]
            offset_x, offset_y = offsets[kpt_idx, :2]
            hm_x[kpt_idx, int(kx), int(ky)] = offset_x
            hm_y[kpt_idx, int(kx), int(ky)] = offset_y
    return hm_x, hm_y


def visualize_maps(bbox, kpts, input_size, output_size, sigma):
    hm_center = gen_center_heatmap(bbox, input_size, output_size, sigma)
    hm_kpt = gen_kpt_heatmap(bbox, kpts, input_size, output_size, sigma)
    hm_reg_x, hm_reg_y = gen_kpt_regression(bbox, kpts, input_size, output_size)
    hm_offset_x, hm_offset_y = gen_offset_regression(bbox, kpts, input_size, output_size)
    cv2.namedWindow('hm_center')
    cv2.imshow('hm_center', hm_center[0])
    cv2.namedWindow('hm_kpt')
    cv2.imshow('hm_kpt', hm_kpt[0])
    center = np.where(hm_reg_x[0, :, :] != 0)
    cx, cy = center[1][0], center[0][0]
    reg_vis = np.zeros(output_size)
    for kpt_idx in range(len(hm_reg_x)):
        kx = int(cx + hm_reg_x[kpt_idx, cy, cx])
        ky = int(cy + hm_reg_y[kpt_idx, cy, cx])
        reg_vis = cv2.arrowedLine(reg_vis, (cx, cy), (kx, ky), 
                                  color=(255, 255, 255), thickness=1)
    cv2.namedWindow('hm_reg')
    cv2.imshow('hm_reg', reg_vis)
    sum_offset_x = np.sum(hm_offset_x, axis=0)
    sum_offset_y = np.sum(hm_offset_y, axis=0)
    zero_map = np.zeros_like(sum_offset_x)
    offset_bgr = np.stack((sum_offset_x, sum_offset_y, zero_map), axis=2)
    offset_bgr = np.clip(offset_bgr * 255, 0, 255).astype(np.uint8)
    cv2.namedWindow('hm_offset')
    cv2.imshow('hm_offset', offset_bgr)
    cv2.waitKey(0)


if __name__ == "__main__":
    bbox = [85, 27, 470, 584]
    input_size = (640, 640)
    output_size = (48, 48)
    sigma = 2
    kpts = [555, 599, 1, 328, 602, 1, 85, 613, 1,
        123, 542, 1, 339, 579, 1, 550, 583, 1,
        104, 578, 1, 131, 303, 1, 130, 294, 1,
        128, 146, 1, 273, 29, 1, 243, 190, 1,
        110, 305, 1, 151, 300, 1, 244, 190, 1,
        255, 72, 1]
    kpts = np.array(kpts).reshape(-1, 3)
    visualize_maps(bbox, kpts, input_size, output_size, sigma)

