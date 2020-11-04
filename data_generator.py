# %%
import math
import os

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence

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


class InconsistentDatasetError(Exception):
    pass


class DOTASequence(Sequence):
    def __init__(self, img_path, annot_path, augmenter=None, batch_size=5):
        self.augmenter = augmenter
        self.img_path = img_path
        self.annot_path = annot_path
        self.batch_size = batch_size
        self.images = [x[:-4] for x in os.listdir(self.img_path) if x.endswith(".png")]

        annotations = {
            x[:-4] for x in os.listdir(self.annot_path) if x.endswith(".txt")
        }
        images_missing_annotation = set(self.images) - annotations
        if len(images_missing_annotation) != 0:
            raise InconsistentDatasetError(
                "Images missing annotations: " + ", ".join(images_missing_annotation)
            )

        # Load all annotations into memory
        self.annotations = {}
        for img_name in self.images:
            file_path = os.path.join(self.annot_path, img_name + ".txt")
            # Test for no annotations
            with open(file_path) as f:
                for i, l in enumerate(f):
                    if i >= 2:
                        self.annotations[img_name] = np.loadtxt(
                            file_path,
                            skiprows=2,
                            converters={8: lambda s: labels_to_idx[s.decode("utf-8")]},
                        )
                        break

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for img_name in batch_images:
            if self.augmenter == None:
                batch_x.append(
                    cv2.imread(os.path.join(self.img_path, img_name + ".png"))
                )
                batch_y.append(np.array(self.annotations[img_name]))
            else:
                x, y = self.augmenter(
                    cv2.imread(os.path.join(self.img_path, img_name + ".png")),
                    np.array(self.annotations[img_name]),
                )
                batch_x.append(x)
                batch_y.append(y)

        return np.asarray(batch_x), np.asarray(batch_y)


class SegmentationSequence(DOTASequence):
    def __getitem__(self, idx):
        imgs, bboxes = super().__getitem__(idx)
        mask_batch = []
        for img, bbox_set in zip(imgs, bboxes):
            mask = np.zeros(
                (img.shape[0], img.shape[1], len(labels_to_idx)), dtype=np.uint8
            )
            for bbox in bbox_set:
                box_class = math.floor(bbox[8])
                bbox = bbox[:8].reshape((4, 2))

                left = np.min(bbox, axis=0)
                right = np.max(bbox, axis=0)
                x = np.arange(math.ceil(left[0]), math.floor(right[0]) + 1)
                y = np.arange(math.ceil(left[1]), math.floor(right[1]) + 1)
                xv, yv = np.meshgrid(x, y, indexing="xy")
                points = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

                path = matplotlib.path.Path(bbox)
                bbox_mask = path.contains_points(points)
                bbox_mask.shape = xv.shape
                mask[
                    math.ceil(left[1]) : math.ceil(right[1]) + 1,
                    math.ceil(left[0]) : math.ceil(right[0]) + 1,
                    box_class,
                ] += bbox_mask
            mask_batch.append(np.where(mask > 0, 1, 0))
        imgs_padded = []
        for img in imgs:
            img_pad = np.zeros(
                (
                    math.ceil(img.shape[0] / 32.0) * 32,
                    math.ceil(img.shape[1] / 32.0) * 32,
                    3,
                )
            )
            img_pad[: img.shape[0], : img.shape[1]] = img
            imgs_padded.append(img_pad)

        masks_padded = []
        for mask in mask_batch:
            mask_pad = np.zeros(
                (
                    math.ceil(mask.shape[0] / 32.0) * 32,
                    math.ceil(mask.shape[1] / 32.0) * 32,
                    16,
                )
            )
            mask_pad[: mask.shape[0], : mask.shape[1]] = mask
            masks_padded.append(mask_pad)

        return np.asarray(imgs_padded), np.asarray(masks_padded)


if __name__ == "__main__":
    sequence = SegmentationSequence(
        ".\\data\\train\\images", ".\\data\\train\\annotations"
    )
# %%
