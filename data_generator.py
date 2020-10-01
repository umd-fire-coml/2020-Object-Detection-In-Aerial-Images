import os

import cv2
import numpy as np
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
    "soccer-ball-field-and-swimming-pool": 13,
}


class InconsistentDatasetError(Exception):
    pass


class DOTASequence(Sequence):
    def __init__(self, img_path, annot_path, augmenter=lambda x: x, batch_size=5):
        self.augmenter = augmenter
        self.img_path = img_path
        self.annot_path = annot_path
        self.batch_size = batch_size
        self.images = [x[:-4] for x in os.listdir(self.img_path) if x.endswith(".png")]
        if annot_path:
            annotations = {
                x[:-4] for x in os.listdir(self.annot_path) if x.endswith(".txt")
            }
            images_missing_annotation = set(self.images) - annotations
            if len(images_missing_annotation) != 0:
                raise InconsistentDatasetError(
                    "Images missing annotations: " + " ".join(images_missing_annotation)
                )

    def __len__(self):
        return len(self.images) / self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for img in batch_images:
            batch_x.append(
                self.augmenter(cv2.imread(os.path.join(self.img_path, img + ".png")))
            )
            batch_y.append(
                np.loadtxt(
                    os.path.join(self.annot_path, img + ".txt"),
                    skiprows=2,
                    converters={8: lambda s: labels_to_idx[s.decode("utf-8")]},
                )
            )

        return np.asarray(batch_x), np.asarray(batch_y)
