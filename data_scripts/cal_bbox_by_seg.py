import cv2
from os.path import join
import os
import numpy as np
from tqdm import tqdm

dir = './datasets/Open_Images/'
mode = 'test'

image_dir = join(dir, 'images', mode)
seg_dir = join(dir, 'segs', mode)

files = os.listdir(seg_dir)

data_dict = {}

for file in tqdm(files):
    seg_path = join(seg_dir, file)
    image_path = join(image_dir, file.split('_')[0] + '.jpg')
    seg = cv2.imread(seg_path)
    image = cv2.imread(image_path)
    seg = cv2.resize(seg, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    seg = seg[:, :, 0]

    # obtain contours point set： contours
    contours = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    if len(contours) > 1:
        cntr = np.vstack(contours)
    elif len(contours) == 1:
        cntr = contours[0]
    else:
        continue

    if len(cntr) < 2:
        continue

    hs, he = np.min(cntr[:, :, 1]), np.max(cntr[:, :, 1])
    ws, we = np.min(cntr[:, :, 0]), np.max(cntr[:, :, 0])

    h, w = seg.shape

    if (he - hs) % 2 == 1 and (he + 1) <= h:
        he = he + 1
    if (he - hs) % 2 == 1 and (hs - 1) >= 0:
        hs = hs - 1
    if (we - ws) % 2 == 1 and (we + 1) <= w:
        we = we + 1
    if (we - ws) % 2 == 1 and (ws - 1) >= 0:
        ws = ws - 1

    if he - hs < 2 or we - ws < 2:
        continue

    data_dict[file] = [cntr, hs, he, ws, we]

np.save(join(dir, 'segs', f'{mode}_bbox_dict.npy'), data_dict)