import random

import albumentations as A
import cv2
import numpy as np


def augment(image, mask):

    # N.B. while the albumentations library is convenient, it lacks any way to
    # represent non-horizontal bounding boxes, so those have to be dealt with manually
    transforms = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    if random.randint(0, 2) == 0:
        # Flip horizontally with 1/3 probability
        image = cv2.flip(image, 1)

        mask = cv2.flip(mask, 1)

    if random.randint(0, 2) == 0:
        # Flip vertically with 1/3 probability
        image = cv2.flip(image, 0)

        mask = cv2.flip(mask, 0)

    if random.randint(0, 1) == 0:
        # rotate a random increment of 90 degrees with 1/2 probability
        turns = random.randint(1, 3)
        if turns == 1:
            # turn 90 clockwise
            image = np.transpose(image, (1, 0, 2))
            image = cv2.flip(image, 1)

            mask = np.transpose(mask, (1, 0, 2))
            mask = cv2.flip(mask, 1)

        elif turns == 2:
            # turn 180
            image = cv2.flip(image, -1)

            mask = cv2.flip(mask, -1)

        else:
            # turn 270
            image = np.transpose(image, (1, 0, 2))
            image = cv2.flip(image, 0)

            mask = np.transpose(mask, (1, 0, 2))
            mask = cv2.flip(mask, 0)


    # Cropping will be handled elsewhere; saving this just in case
    """
    if (random.randint(0,1) == 0):
        #Crop randomly with 1/2 probability
        xmin = random.randint(0,int(width/4))
        xmax = random.randint(int(width*0.75),width)
        ymin = random.randint(0,int(height/4))
        ymax = random.randint(int(height*0.75),height)
        #Crops up to 1/4 of the image off on all sides

        cut_boxes = []
        for i in range(annots.shape[0]):
            box = annots[i]
            if (box[0] < xmin or box[2] < xmin or box[4] < xmin or box[6] < xmin):
                cut_boxes.append(i)
            if (box[1] < ymin or box[3] < ymin or box[5] < ymin or box[7] < ymin):
                cut_boxes.append(i)
            if (box[0] > xmax or box[2] > xmax or box[4] > xmax or box[6] > xmax):
                cut_boxes.append(i)
            if (box[1] > ymax or box[3] > ymax or box[5] > ymax or box[7] > ymax):
                cut_boxes.append(i)

            #Remove bounding boxes that get cut off

            box[0] = box[0] - xmin
            box[2] = box[2] - xmin
            box[4] = box[4] - xmin
            box[6] = box[6] - xmin

            box[1] = box[1] - ymin
            box[3] = box[3] - ymin
            box[5] = box[5] - ymin
            box[7] = box[7] - ymin
            #Adjust remaining bounding boxes according to crop

        annots = np.delete(annots,cut_boxes,0)

        transforms.append(A.Crop(x_min = xmin, y_min = ymin, x_max = xmax, y_max = ymax, p=1.0))
        height = ymax - ymin
        width = xmax - xmin
    """

    # transforms.append(A.RandomBrightnessContrast(p=0.5))
    # Randomly shift brightness and contrast with 50% probability

    transforms.append(A.CLAHE(p=1))
    # Apply Contrast Limited Adaptive Histogram Equalization (???) with 50% probability

    augment = A.Compose(transforms)

    return augment(image=image)["image"], mask
