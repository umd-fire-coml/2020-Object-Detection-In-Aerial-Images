#Iterates through all files in data, checking if they're pngs or corrupted

import os
from PIL import Image

badFiles = []
for file in os.listdir("./data"):
    if not file.endswith(".png"):
        badFiles.append(file)
    else:
        try:
            img = Image.open("./data/" + file)
            img.verify()
        except:
            badFiles.append(file + "CORRUPT")
if len(badFiles) == 0:
    print("All good!")
else:
    print("Bad things: " + badFiles)