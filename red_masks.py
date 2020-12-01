# %%
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image as im

from data_generator import onehot_to_rgb, rgb_to_onehot

#%% Color maps
# new colormap
colormap = [
    (0, 0, 0),  # void
    (230, 20, 230),  # plane
    (65, 26, 125),  # ship
    (15, 107, 226),  # storage tank
    (0, 63, 0),  # baseball diamond
    (0, 63, 127),  # tennis court
    (0, 63, 193),  # basketball court
    (229, 37, 37),  # ground-track-field
    (22, 190, 209),  # harbor
    (159, 22, 209),  # bridge
    (13, 169, 138),  # large vehicle
    (13, 174, 29),  # small vehicle
    (74, 30, 92),  # helicopter
    (119, 96, 46),  # roundabout
    (127, 191, 0),  # Soccer field
    (255, 0, 0),  # Swimming pool
    (124, 78, 0),  # Container Crane
]

# original colormap
og_colormap = [
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

#%% function to convert image
def use_red_channel(img_dir):
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    onehot = rgb_to_onehot(image, og_colormap)
    return onehot_to_rgb(onehot, colormap)


#%% test cell
# output = use_red_channel('data/train/masks/P0980_instance_color_RGB.png')
# cv2.imwrite('example.png', output)
#%%
directories = ["data/train/masks/", "data/validation/masks"]

# iterating directories
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Make use of red channel
            fulldir = directory + filename
            image = use_red_channel(fulldir)
            cv2.imwrite(fulldir, image)
