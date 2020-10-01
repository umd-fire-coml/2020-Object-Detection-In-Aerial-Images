import cv2
import copy

# takes image as cv2 image (from cv2.imread) and annots as a matrix of annotations

def vertical_flip (image, annots):
    height, width = image.shape[:2]
    augimage = cv2.flip(image, 0)
    
    aug_annots = copy.deepcopy(annots)
    for box in aug_annots:
        box[1] = height - box[1]
        box[3] = height - box[3]
        box[5] = height - box[5]
        box[7] = height - box[7]
        
    return augimage, aug_annots
    
def horizontal_flip (image, annots):
    height, width = image.shape[:2]
    augimage = cv2.flip(image, 1)
    
    aug_annots = copy.deepcopy(annots)
    for box in aug_annots:
        box[0] = width - box[0]
        box[2] = width - box[2]
        box[4] = width - box[4]
        box[6] = width - box[6]
        
    return augimage, aug_annots
    
def rotate (image, annots, angle): # increments of 90 only
    aug_annots = copy.deepcopy(annots)    
    
    if angle == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        for box in aug_annots:
            x1, x2, x3, x4 = box[0], box[2], box[4], box[6]
            y1, y2, y3, y4 = box[1], box[3], box[5], box[7]
            box[0] = -1 * y1
            box[2] = -1 * y2
            box[4] = -1 * y3
            box[6] = -1 * y4
            
            box[1] = x1
            box[3] = x2
            box[5] = x3
            box[7] = x4
    
    else if angle == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
        for box in aug_annots:
            x1, x2, x3, x4 = box[0], box[2], box[4], box[6]
            y1, y2, y3, y4 = box[1], box[3], box[5], box[7]
            box[0] = -1 * y1
            box[2] = -1 * y2
            box[4] = -1 * y3
            box[6] = -1 * y4
            
            box[1] = -1 * x1
            box[3] = -1 * x2
            box[5] = -1 * x3
            box[7] = -1 * x4
    
    else if angle == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for box in aug_annots:
            x1, x2, x3, x4 = box[0], box[2], box[4], box[6]
            y1, y2, y3, y4 = box[1], box[3], box[5], box[7]
            box[0] = y1
            box[2] = y2
            box[4] = y3
            box[6] = y4
            
            box[1] = -1 * x1
            box[3] = -1 * x2
            box[5] = -1 * x3
            box[7] = -1 * x4
    
   return image, aug_annots
   