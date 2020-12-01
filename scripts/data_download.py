# %%
import gdown
import os
from zipfile import ZipFile

# %% setting up data directories
if not(os.path.exists(os.path.join('..', 'data', 'train'))):
    os.makedirs(os.path.join('..', 'data', 'train'))

if not(os.path.exists(os.path.join('..', 'data', 'train', 'annotations_hbb'))):
    os.makedirs(os.path.join('..', 'data', 'train', 'annotations_hbb'))

if not(os.path.exists(os.path.join('..', 'data', 'train', 'annotations'))):
    os.makedirs(os.path.join('..', 'data', 'train', 'annotations'))

if not(os.path.exists(os.path.join('..', 'data', 'test'))):
    os.makedirs(os.path.join('..', 'data', 'test'))

if not(os.path.exists(os.path.join('..', 'data', 'validation'))):
    os.makedirs(os.path.join('..', 'data', 'validation'))

if not(os.path.exists(os.path.join('..', 'data', 'validation', 'annotations_hbb'))):
    os.makedirs(os.path.join('..', 'data', 'validation', 'annotations_hbb'))

if not(os.path.exists(os.path.join('..', 'data', 'validation', 'annotations'))):
    os.makedirs(os.path.join('..', 'data', 'validation', 'annotations'))


# %% training set
# google drive download urls
urls = ['https://drive.google.com/uc?export=download&id=1zb_kEOXTtpsMuWAtuTtMIPLVUSpjGPPw',
'https://drive.google.com/uc?export=download&id=1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v', 
'https://drive.google.com/uc?export=download&id=1pEmwJtugIWhiwgBqOtplNUtTG2T454zn',
'https://drive.google.com/uc?export=download&id=1-vLCMhIW9CV2cmCPPBbDR9_hdecf5bLb',
'https://drive.google.com/uc?export=download&id=12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX']

outputs = [os.path.join('..', 'data', 'train', 'part1.zip'), os.path.join('..', 'data', 'train', 'part2.zip'),
            os.path.join('..', 'data', 'train', 'part3.zip'), os.path.join('..', 'data', 'train', 'annotations_hbb', 'annotations.zip'),
            os.path.join('..', 'data', 'train', 'annotations', 'annotations.zip')]

for i in range(len(urls)):
    if not(os.path.exists(outputs[i])):
        gdown.download(urls[i],outputs[i], quiet = False)



# %% validation
urls = ['https://drive.google.com/uc?export=download&id=1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP',
'https://drive.google.com/uc?export=download&id=1XDWNx3FkH9layL8jVUkEHJ_-CY8K4zse',
'https://drive.google.com/uc?export=download&id=1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC']

outputs = ['data/validation/part1.zip', 'data/validation/annotations_hbb/annotations.zip',
'data/validation/annotations/annotations.zip']

for i in range(len(urls)):
    outputs[i] = os.path.normpath(os.path.join('..', outputs[i]))
    if not(os.path.exists(outputs[i])):
        gdown.download(urls[i],outputs[i], quiet = False)



# %% test
urls = ['https://drive.google.com/uc?export=download&id=1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK',
'https://drive.google.com/uc?export=download&id=1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv',
'https://drive.google.com/uc?export=download&id=1nQokIxSy3DEHImJribSCODTRkWlPJLE3']

outputs = ['data/test/part1.zip', 'data/test/part2.zip', 'data/test/test_info.json']

for i in range(len(urls)):
    outputs[i] = os.path.normpath(os.path.join('..', outputs[i]))
    if not(os.path.exists(outputs[i])):
        gdown.download(urls[i],outputs[i], quiet = False)

# %% Unzipping

zip_paths = ['data/train/part1.zip', 'data/train/part2.zip', 'data/train/part3.zip',
'data/train/annotations_hbb/annotations.zip', 'data/validation/part1.zip', 
'data/validation/annotations_hbb/annotations.zip', 'data/test/part1.zip', 'data/test/part2.zip',
'data/train/annotations/annotations.zip','data/validation/annotations/annotations.zip']

outputs = ['data/train/', 'data/train/', 'data/train/', 'data/train/annotations_hbb/',
'data/validation/', 'data/validation/annotations_hbb/', 'data/test/', 'data/test/','data/train/annotations/',
'data/validation/annotations/']

for i in range(len(zip_paths)):
    zip_paths[i] = os.path.normpath(os.path.join('..', zip_paths[i]))
    outputs[i] = os.path.normpath(os.path.join('..', outputs[i]))
    if os.path.exists(zip_paths[i]):
        with ZipFile(zip_paths[i], 'r') as zip_ref:
            zip_ref.extractall(outputs[i])
    else:
        print("Missing file: ", zip_paths[i], "\nDownload manually if google drive access was denied")

# %%
