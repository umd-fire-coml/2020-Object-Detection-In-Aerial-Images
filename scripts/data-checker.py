#Iterates through all files in data

import os
from PIL import Image

bad_files = []
paths = ['train/images', 'test/images', 'validation/images',
        'train/annotations_hbb', 'train/annotations', 
        'validation/annotations_hbb', 'validation/annotations']
file_end = '.png'

# Goes through expected file structure, checking if file ends are correct, and pngs for corruption
for path_bad in paths:
    path = os.path.join('..', 'data', os.path.normpath(path_bad))
    if path == os.path.normpath('data/train/annotations_hbb'):
        file_end = '.txt'
    for file in os.listdir(path):
        if not file.endswith(file_end):
            bad_files.append(file)
            if file.endswith('.zip'):
                os.remove(os.path.join(path, file))
                print("Removed: " + file)
        elif file_end == '.png':
            try:
                with Image.open(os.path.join(path, file)) as img:
                    img.verify()
            except:
                bad_files.append(file + " CORRUPT")
if len(bad_files) == 0:
    print("All good!")
else:
    print("Bad things: ")
    for p in bad_files:
        print(p)
