import json
import os
import numpy
import cv2

# Takes a filepath to the .json containing annotations, returns a dictionary mapping category ids to the total pixels of that category
def get_absolute_occ(filein):
    out = dict()
    for i in range(15):
        out[i+1] = 0
    f = open(filein,"r")
    d = json.loads(f.read())
    for x in d['annotations']:
        out[x['category_id']] += x['area']
    return out

# Takes a directory of images, returns total pixels
def get_total_pixels(imgdir):
    out = 0
    for f in os.listdir(imgdir):
        if f.endswith(".png"):
            im = cv2.imread(os.path.join(imgdir,f))
            out += (im.shape[0]*im.shape[1])
    return out

def get_relative_occ(filein,imgdir):
    ab = get_absolute_occ(filein)
    tot = get_total_pixels(imgdir)
    out = dict()
    for x in ab.keys():
        out[x] = ab[x]/tot
    return out