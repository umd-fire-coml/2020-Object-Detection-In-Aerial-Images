import os

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

labels_to_idx = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
    "container-crane": 15,
}

# It'd be most convenient to use the save_all_masks function, passing in the image path, annotation path, and target path to save the masks.
# The functions use numpy's convenient file-saving feature to save the mask arrays as .npy files
# load_mask is intended to ease the process of loading masks from disk

def get_mask(img, bbox_set):
    
    mask = np.zeros(
        (img.shape[0], img.shape[1], 16), dtype = np.uint8)
    for bbox in bbox_set:
        try:
            box_class = bbox[8]
        except IndexError as e:
            print(img_name)
            print(bbox)
            raise e

        bbox = bbox[:8].reshape((4, 2))

        left = np.min(bbox, axis=0)
        right = np.max(bbox, axis=0)
        x = np.arange(left[0], right[0] + 1)
        y = np.arange(left[1], right[1] + 1)
        xv, yv = np.meshgrid(x, y, indexing="xy")
        points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        path = matplotlib.path.Path(bbox)
        bbox_mask = path.contains_points(points)
        bbox_mask.shape = xv.shape
        try:
            mask[
                left[1] : right[1] + 1,
                left[0] : right[0] + 1,
                box_class,
            ] += bbox_mask
        except ValueError as e:
            pass
    return mask

def save_mask(img_path, annot_path, img_name, mask_path):
    img = cv2.imread(os.path.join(img_path, img_name + '.png'))
    path = os.path.join(annot_path, img_name + ".txt")
    with open(path) as f:
        for i, l in enumerate(f):
            if i >= 2:
                bboxes = np.loadtxt(
                    path,
                    skiprows=2,
                    converters={8: lambda s: labels_to_idx[s.decode("utf-8")]},
                ).astype(np.uint16)
                bboxes.shape = (-1, 10)
                bboxes[:, :8] -= 1
                break
    
    mask = get_mask(img, bboxes)
    save_path = os.path.join(mask_path, img_name + ".npy")
    np.save(save_path, mask)
        
def save_all_masks(img_path, annot_path, mask_path):
    image_names = [x[:-4] for x in os.listdir(img_path) if x.endswith(".png")]
    annotations = {x[:-4] for x in os.listdir(annot_path) if x.endswith(".txt")}
    if not(os.path.exists(mask_path)):
        os.makedirs(mask_path)
    for image_name in image_names:
        
        if (image_name not in annotations) or os.path.exists(os.path.join(mask_path,image_name + ".npy")):
            continue
        
        print("Saving mask for " + image_name) 
        save_mask(img_path,annot_path,image_name,mask_path)
        
def load_mask(mask_path, mask_name):
    load_path = os.path.join(mask_path, mask_name + ".npy")
    return np.load(load_path)
