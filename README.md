# 2020-Object-Detection-In-Aerial-Images

## Setup Instructions (Contributors)
run `setup.sh`

Note: Windows users run `setup.sh` in administrator Git Bash. Windows Subsystem Linux (2) not recommended due to issues with GPU passthrough.

## Setup Instructions (Users)

run `conda env create --file environment.yml`

If there are problems with the environment afterwards, run the [environment checker](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/environment_checker.py).

## Product Description

This repo detects rotated and cluttered objects in aerial images. The model itself is a convoultional neural network using several groups of convolutional/deconvolutional and maxpooling layers. We use rotation augmentation to further account for the various rotations objects may be found in.

## Video Demonstration

A demonstration can be found [here](https://youtu.be/tSZCD2kxmtQ).

## Results

Visualized results notebook [here](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/Visualizer.ipynb).

## Directory Guide

### AccDemo.ipynb
A simple test of accuracy.py.

### Visualizer.ipynb
Visualizes final product data.

### accuracy.py
Accuracy function. Sourced from [here](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2). [1]

### augmentation.py
Augmentation function. Rotates images to generate more data.

### data-checker.py
Verifies integrity of data. Deletes duplicate zip files.

### data_download.py 
Downloads dataset. The code uses the [DOTA](https://captain-whu.github.io/DOTA/index.html) dataset. [2]

### data_generator.py
Prepares data to be fed into the model.

### demo_notebook.ipynb
Demonstrates final trained model.

### environment.yml
Environment file necessary for running everything else.

### environment_checker.py
Verifies integrity of current environment, checks if versions are up to date.

### masks.py
Mask functions, allowing for saving of masks.

### model_builder.py
Builds the model, creating an untrained model.

### red_masks.py
Changes mask coloring to more appropriate color.

### train.py
Model training functions, given a model created by model_builder.py. Includes tversky loss function sourced from [here](https://arxiv.org/abs/1706.05721) [3]

## Training Guide

The data can be downloaded by running [data_download.py](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/data_download.py). 
Afterwards, run [data-checker.py](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/data-checker.py) to verify files.
Then run [model_builder.py](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/data-checker.py) to begin training.

## Sources

[1] Ekin Tiu. 2019. Metrics to Evaluate Your Semantic Segmentation Model. (August 2019). Retrieved November 14, 2020 from https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

[2] Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei. 2018. DOTA: A Large-Scale Dataset for Object Detection in Aerial Images. The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[3] Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour. 2017. Tversky loss function for image segmentation using 3D fully convolutional deep networks. arXiv:1706.05721. Retrieved from https://arxiv.org/abs/1706.05721.
