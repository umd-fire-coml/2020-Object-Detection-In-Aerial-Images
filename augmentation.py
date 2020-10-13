import cv2
import numpy as np
import albumentations as A
import random

def augment (image, annots):
    # N.B. while the albumentations library is convenient, it lacks any way to
    # represent non-horizontal bounding boxes, so those have to be dealt with manually
    
    transforms = []
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if (random.randint(0,2) == 0):
        #Flip horizontally with 1/3 probability
        image = cv2.flip(image,1)
        for box in annots:
            box[0] = width - box[0]
            box[2] = width - box[2]
            box[4] = width - box[4]
            box[6] = width - box[6]
    
    if (random.randint(0,2) == 0):
        #Flip vertically with 1/3 probability
        cv2.flip(image,0)
        for box in annots:
            box[1] = height - box[1]
            box[3] = height - box[3]
            box[5] = height - box[5]
            box[7] = height - box[7]
    
    if (random.randint(0,1) == 0):
        #rotate a random increment of 90 degrees with 1/2 probability
        turns = random.randint(1,3)
        if (turns == 1):
            #turn 90 clockwise
            image = np.transpose(image, (1,0,2))
            image = cv2.flip(image, 1)
            for box in annots:
                x1, x2, x3, x4 = box[0], box[2], box[4], box[6]
                y1, y2, y3, y4 = box[1], box[3], box[5], box[7]
                box[0] = height - y1
                box[2] = height - y2
                box[4] = height - y3
                box[6] = height - y4
                
                box[1] = x1
                box[3] = x2
                box[5] = x3
                box[7] = x4
                
            temp = height
            height = width
            width = temp
                
        elif (turns == 2):
            #turn 180
            image = cv2.flip(image, -1)
            for box in annots:
                x1, x2, x3, x4 = box[0], box[2], box[4], box[6]
                y1, y2, y3, y4 = box[1], box[3], box[5], box[7]
                box[0] = width - x1
                box[2] = width - x2
                box[4] = width - x3
                box[6] = width - x4
                
                box[1] = height - y1
                box[3] = height - y2
                box[5] = height - y3
                box[7] = height - y4
        else:
            #turn 270
            image = np.transpose(image, (1,0,2))
            image = cv2.flip(image, 0)
            for box in annots:
                x1, x2, x3, x4 = box[0], box[2], box[4], box[6]
                y1, y2, y3, y4 = box[1], box[3], box[5], box[7]
                box[0] = y1
                box[2] = y2
                box[4] = y3
                box[6] = y4
                
                box[1] = width - x1
                box[3] = width - x2
                box[5] = width - x3
                box[7] = width - x4
                
            temp = height
            height = width
            width = temp
    
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
    
    transforms.append(A.RandomBrightnessContrast(p=0.5))
    # Randomly shift brightness and contrast with 50% probability
    
    augment = A.Compose(transforms)
    return augment(image=image)['image'], annots
    