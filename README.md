# 2020-Object-Detection-In-Aerial-Images

## Setup Instructions (Contributors)
run `setup.sh`

Note: Windows users run `setup.sh` in administrator Git Bash. Windows Subsystem Linux (2) not recommended due to issues with GPU passthrough.

## Setup Instructions (Users)

run `conda env create --file environment.yml`

If there are problems with the environment afterwards, run the [environment checker](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/environment_checker.py).

## Main Features

This repo detects rotated and cluttered objects in aerial images. We use rotation augmentation to further account for the various rotations objects may be found in.

## Dataset

The code uses the [DOTA](https://captain-whu.github.io/DOTA/index.html) dataset. This can be downloaded by running [data_download.py](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/data_download.py). Afterwards, run [data-checker.py](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/data-checker.py) to verify files.
Data can be visualized using the visualizing [notebook](https://github.com/umd-fire-coml/2020-Object-Detection-In-Aerial-Images/blob/master/Visualizer.ipynb).
