# %%
import os
import cv2
# %% test cell
#flag = os.path.exists('data/train/masks/')
#image = cv2.imread('data/train/masks/P0001_instance_color_RGB.png')
#
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imshow('test', image)
#cv2.imwrite('red_example.png', image)
#cv2.waitKey()


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
