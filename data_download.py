# %%
import gdown
import os
from zipfile import ZipFile

# %% setting up data directories
if not(os.path.exists('data/train')):
    os.makedirs('data/train')

if not(os.path.exists('data/train/annotations_hbb/')):
    os.makedirs('data/train/annotations_hbb/')

if not(os.path.exists('data/train/annotations/')):
    os.makedirs('data/train/annotations/')

if not(os.path.exists('data/test')):
    os.makedirs('data/test')

if not(os.path.exists('data/validation')):
    os.makedirs('data/validation')

if not(os.path.exists('data/validation/annotations_hbb/')):
    os.makedirs('data/validation/annotations_hbb/')

if not(os.path.exists('data/validation/annotations/')):
    os.makedirs('data/validation/annotations/')


# %% training set
# google drive download urls
urls = ['https://drive.google.com/uc?export=download&id=1zb_kEOXTtpsMuWAtuTtMIPLVUSpjGPPw',
'https://drive.google.com/uc?export=download&id=1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v', 
'https://drive.google.com/uc?export=download&id=1pEmwJtugIWhiwgBqOtplNUtTG2T454zn',
'https://drive.google.com/uc?export=download&id=1-vLCMhIW9CV2cmCPPBbDR9_hdecf5bLb',
'https://drive.google.com/uc?export=download&id=12uPWoADKggo9HGaqGh2qOmcXXn-zKjeX']

outputs = ['data/train/part1.zip', 'data/train/part2.zip', 'data/train/part3.zip', 
'data/train/annotations_hbb/annotations.zip', 'data/train/annotations/annotations.zip']

for i in range(len(urls)):
    gdown.download(urls[i],outputs[i], quiet = False)



# %% validation
urls = ['https://drive.google.com/uc?export=download&id=1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP',
'https://drive.google.com/uc?export=download&id=1XDWNx3FkH9layL8jVUkEHJ_-CY8K4zse',
'https://drive.google.com/uc?export=download&id=1FkCSOCy4ieNg1UZj1-Irfw6-Jgqa37cC']

outputs = ['data/validation/validation.zip', 'data/validation/annotations_hbb/annotations.zip',
'data/validation/annotations/annotations.zip']

for i in range(len(urls)):
    gdown.download(urls[i],outputs[i], quiet = False)



# %% test
urls = ['https://drive.google.com/uc?export=download&id=1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK',
'https://drive.google.com/uc?export=download&id=1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv',
'https://drive.google.com/uc?export=download&id=1nQokIxSy3DEHImJribSCODTRkWlPJLE3']

outputs = ['data/test/part1.zip', 'data/test/part2.zip', 'data/test/test_info.json']

for i in range(len(urls)):
    gdown.download(urls[i],outputs[i], quiet = False)

# %% Unzipping

zip_paths = ['data/train/part1.zip', 'data/train/part2.zip', 'data/train/part3.zip',
'data/train/annotations_hbb/annotations.zip', 'data/validation/validation.zip', 
'data/validation/annotations_hbb/annotations.zip', 'data/test/part1.zip', 'data/test/part2.zip',
'data/train/annotations/annotations.zip','data/validation/annotations/annotations.zip']

outputs = ['data/train/', 'data/train/', 'data/train/', 'data/train/annotations_hbb/',
'data/validation/', 'data/validation/annotations_hbb/', 'data/test/', 'data/test/','data/train/annotations/',
'data/validation/annotations/']

for i in range(len(zip_paths)):
    with ZipFile(zip_paths[i], 'r') as zip_ref:
        zip_ref.extractall(outputs[i])
