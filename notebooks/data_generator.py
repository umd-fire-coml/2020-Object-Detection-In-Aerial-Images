# %%
import os
import random
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence

labels_to_idx = {
    "void": 0,
    "plane": 1,
    "ship": 2,
    "storage-tank": 3,
    "baseball-diamond": 4,
    "tennis-court": 5,
    "basketball-court": 6,
    "ground-track-field": 7,
    "harbor": 8,
    "bridge": 9,
    "large-vehicle": 10,
    "small-vehicle": 11,
    "helicopter": 12,
    "roundabout": 13,
    "soccer-ball-field": 14,
    "swimming-pool": 15,
    "container-crane": 16,
}

idx_to_rgb = [
    (0, 0, 0),
    (0, 127, 255),
    (0, 0, 63),
    (0, 63, 63),
    (0, 63, 0),
    (0, 63, 127),
    (0, 63, 191),
    (0, 63, 255),
    (0, 100, 155),
    (0, 127, 63),
    (0, 127, 127),
    (0, 0, 127),
    (0, 0, 191),
    (0, 191, 127),
    (0, 127, 191),
    (0, 0, 255),
]

rgb_to_idx = {v: k for k, v in enumerate(idx_to_rgb)}


def rgb_to_onehot(rgb_image, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2] + (num_classes,)
    encoded_image = np.zeros(shape, dtype=np.float32)
    for i, cls in enumerate(colormap):
        encoded_image[:, :, i] = np.all(
            rgb_image.reshape((-1, 3)) == colormap[i], axis=1
        ).reshape(shape[:2])
    return np.asarray(encoded_image)


def onehot_to_rgb(onehot, colormap):
    """Function to decode encoded mask labels
    Inputs:
        onehot - one hot encoded image matrix (height x width x num_classes)
        colormap - dictionary of color to label id
    Output: Decoded RGB image (height x width x 3)
    """
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros(onehot.shape[:2] + (3,))
    for k in colormap.keys():
        output[single_layer == k] = colormap[k]
    return np.uint8(output)


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
                        self.annotations[img_name].shape = (-1, 10)
                        self.annotations[img_name][:, :8] -= 1
                        break

            if img_name not in self.annotations.keys():
                self.annotations[img_name] = []

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


class SegmentationSequence(Sequence):
    def __init__(
        self,
        img_path,
        mask_path,
        augmenter=None,
        batch_size=5,
        output_img_size=(480, 480),
        steps_per_epoch=1000,
    ):
        self.img_path = img_path
        self.mask_path = mask_path
        self.augmenter = augmenter
        self.batch_size = batch_size
        self.output_img_size = output_img_size
        self.steps_per_epoch = steps_per_epoch
        self.generator = self.batch_generator()

        self.image_names = [
            x[:-4] for x in os.listdir(self.img_path) if x.endswith(".png")
        ]

        masks = {
            x[:-23]
            for x in os.listdir(self.mask_path)
            if x.endswith("_instance_color_RGB.png")
        }
        images_missing_masks = set(self.image_names) - masks
        if len(images_missing_masks) != 0:
            raise InconsistentDatasetError(
                "Images missing annotations: " + ", ".join(images_missing_masks)
            )

    def __len__(self):
        return self.steps_per_epoch

    def batch_generator(self):
        start_pos = (0, 0)
        while True:
            for img_name in self.image_names:
                img = cv2.imread(os.path.join(self.img_path, img_name + ".png"))
                mask = cv2.imread(
                    os.path.join(self.mask_path, img_name + "_instance_color_RGB.png")
                )
                if self.augmenter:
                    img, mask = self.augmenter(img, mask)

                mask = rgb_to_onehot(mask, idx_to_rgb)

                for x in range(start_pos[0], img.shape[1], self.output_img_size[0]):
                    for y in range(start_pos[1], img.shape[0], self.output_img_size[1]):
                        img_chunk = img[
                            y : min(y + self.output_img_size[1], img.shape[0]),
                            x : min(x + self.output_img_size[0], img.shape[1]),
                        ]
                        mask_chunk = mask[
                            y : min(y + self.output_img_size[1], mask.shape[0]),
                            x : min(x + self.output_img_size[0], mask.shape[1]),
                        ]
                        if 1 in mask_chunk[:, :, 1:]:
                            yield img_chunk, mask_chunk

            random.shuffle(self.image_names)

    def __getitem__(self, idx):
        imgs, masks = zip(*[next(self.generator) for i in range(self.batch_size)])

        imgs = list(imgs)
        masks = list(masks)

        for i, img in enumerate(imgs):
            if img.shape != (self.output_img_size[1], self.output_img_size[0], 3):
                img_pad = np.zeros(
                    (self.output_img_size[1], self.output_img_size[0], 3),
                    dtype=np.uint8,
                )
                img_pad[: img.shape[0], : img.shape[1]] = img
                imgs[i] = img_pad

        for i, mask in enumerate(masks):
            if mask.shape != (
                self.output_img_size[1],
                self.output_img_size[0],
                len(labels_to_idx) - 1,
            ):
                mask_pad = np.zeros(
                    (
                        self.output_img_size[1],
                        self.output_img_size[0],
                        len(labels_to_idx) - 1,
                    )
                )
                mask_pad[: mask.shape[0], : mask.shape[1]] = mask
                masks[i] = mask_pad

        return np.asarray(imgs), np.asarray(masks)


if __name__ == "__main__":
    sequence = SegmentationSequence(os.path.normpath(".\\data\\train\\images", ".\\data\\train\\masks"))
    i = 0
    while True:
        imgs, masks = sequence[0]
        if imgs.shape != (5, 480, 480, 3):
            print("Sample {} img incorrect load. Shape {}".format(i, imgs.shape))
            sys.exit()
        if masks.shape != (5, 480, 480, 16):
            print("Sample {} mask incorrect load. Shape {}".format(i, masks.shape))
            sys.exit()

# %%
