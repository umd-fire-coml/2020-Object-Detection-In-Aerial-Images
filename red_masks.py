# %%
import os
import cv2

#%%
directory = 'data/train/masks/'
#iterating directory
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        #Doing BGR to RGB
        fulldir = directory + filename
        image = cv2.imread(directory+filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fulldir, image)
