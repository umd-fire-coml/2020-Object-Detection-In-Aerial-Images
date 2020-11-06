# %%
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
        self.image_names = [
            x[:-4] for x in os.listdir(self.img_path) if x.endswith(".png")
        ]

        annotations = {
            x[:-4] for x in os.listdir(self.annot_path) if x.endswith(".txt")
        }
        images_missing_annotation = set(self.image_names) - annotations
        if len(images_missing_annotation) != 0:
            raise InconsistentDatasetError(
                "Images missing annotations: " + ", ".join(images_missing_annotation)
            )

        # Load all annotations into memory
        self.annotations = {}
        for img_name in self.image_names:
            file_path = os.path.join(self.annot_path, img_name + ".txt")
            # Test for no annotations
            with open(file_path) as f:
                for i, l in enumerate(f):
                    if i >= 2:
                        self.annotations[img_name] = np.loadtxt(
                            file_path,
                            skiprows=2,
                            converters={8: lambda s: labels_to_idx[s.decode("utf-8")]},
                        ).astype(np.uint16)
                        break

    def __len__(self):
        return len(self.image_names) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_names[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
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
    def __init__(
        self,
        img_path,
        annot_path,
        augmenter=None,
        batch_size=5,
        output_img_size=(480, 480),
    ):
        super().__init__(img_path, annot_path, augmenter, batch_size)
        self.output_img_size = output_img_size

    def __getitem__(self, idx):
        def batch_gen():
            start_pos = (0, 0)
            while True:
                for img_name in self.image_names:
                    img = cv2.imread(os.path.join(self.img_path, img_name + ".png"))
                    bbox_set = self.annotations[img_name]

                    if self.augmenter:
                        img, bbox_set = self.augmenter(img, bbox_set)

                    mask = np.zeros(
                        (img.shape[0], img.shape[1], len(labels_to_idx)), dtype=np.uint8
                    )
                    for bbox in bbox_set:
                        box_class = bbox[8]
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
                            print(bbox)
                            print(mask.shape)
                            print(img_name)
                            print(
                                mask[
                                    left[1] : right[1] + 1,
                                    left[0] : right[0] + 1,
                                    box_class,
                                ].shape
                            )
                            print(left)
                            print(right)
                            print(str(left[0]) + ", " + str(right[0]))
                            raise e

                    for x in range(start_pos[0], img.shape[1], self.output_img_size[0]):
                        for y in range(
                            start_pos[1], img.shape[0], self.output_img_size[1]
                        ):
                            yield img[
                                y : min(y + self.output_img_size[1], img.shape[0]),
                                x : min(x + self.output_img_size[0], img.shape[1]),
                            ], mask[
                                y : min(y + self.output_img_size[1], mask.shape[0]),
                                x : min(x + self.output_img_size[0], mask.shape[1]),
                            ]

        generator = batch_gen()
        imgs, masks = zip(*[next(generator) for i in range(self.batch_size)])

        for i, img in enumerate(imgs):
            if img.shape != (self.output_img_size[1], self.output_img_size[0], 3):
                img_pad = np.zeros(
                    (self.output_img_size[1], self.output_img_size[0], 3)
                )
                img_pad[: img.shape[0], : img.shape[1]] = img
                imgs[i] = img_pad

        for i, mask in enumerate(masks):
            if mask.shape != (
                self.output_img_size[1],
                self.output_img_size[0],
                len(labels_to_idx),
            ):
                mask_pad = np.zeros(
                    (
                        self.output_img_size[1],
                        self.output_img_size[0],
                        len(labels_to_idx),
                    )
                )
                mask_pad[: mask.shape[0], : mask.shape[1]] = mask
                masks[i] = mask_pad

        return np.asarray(imgs), np.asarray(masks)


if __name__ == "__main__":
    sequence = SegmentationSequence(
        ".\\data\\train\\images", ".\\data\\train\\annotations"
    )
# %%
