# %% imports
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from data_generator import SegmentationSequence

# %% get random test set image
folder_path = "data/test/images/"


def get_test_img():
    dirs = os.listdir(folder_path)
    file_path = random.choice(dirs)
    img_path = os.path.join(folder_path, file_path)
    image = cv2.imread(img_path)
    return (image, file_path)


#%% getting image
img_data = get_test_img()
image = img_data[0]
img_name = img_data[1]
width = image.shape[0]
height = image.shape[1]

# %% get the model
# will later change this to load our trained weights
def get_model():
    model = tf.keras.applications.DenseNet201(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(width, height, 3),
        pooling=None,
        classes=1000,
    )

    return model


#%% Demonstration
model = get_model()

img_array = np.expand_dims(image, axis=0).astype("float")
prediction = model.predict(img_array)

print("Test Image: ", img_name)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(image)
plt.show()

print("Model Prediction: ")
fig = plt.figure()
ax = fig.add_subplot(111)
prediction = np.argmax(prediction.squeeze(), axis=-1)
ax.imshow(prediction)
plt.show()
