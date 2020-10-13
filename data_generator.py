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
    "soccer-ball-field": 13,
    "swimming-pool": 14,
    "container-crane": 15,
}


class InconsistentDatasetError(Exception):
    pass


class DOTASequence(Sequence):
    def __init__(self, img_path, annot_path, augmenter = None, batch_size=5):
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
                    "Images missing annotations: "
                    + ", ".join(images_missing_annotation)
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
        return len(self.images) / self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for img_name in batch_images:
            if (self.augmenter) == None:
                batch_x.append(cv2.imread(os.path.join(self.img_path, img_name + ".png")))
                batch_y.append(self.annotations[img_name])
            else:
                x, y = self.augmenter(
                    cv2.imread(os.path.join(self.img_path, img_name + ".png")),
                    self.annotations[img_name]
                )
                batch_x.append(x)
                batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)
