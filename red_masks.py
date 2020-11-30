# %%
import os
import cv2

#%%
directories = ["data/train/masks/", "data/validation/masks/"]

# iterating directories
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Doing BGR to RGB
            fulldir = directory + filename
            image = cv2.imread(directory + filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(fulldir, image)
