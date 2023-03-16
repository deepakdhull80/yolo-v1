import os

import wget
from zipfile import ZipFile

import config

# download coco dataset
DATA_FILE = f"{config.dataset_name}.zip"
ANNOTATION_FILE = f"annotations_trainval{config.dataset_name[-4:]}.zip"
DATA_DIR = config.data_base_path

print("downloading dataset")
wget.download(f"http://images.cocodataset.org/zips/{DATA_FILE}",f"{DATA_DIR}/")
wget.download(f"http://images.cocodataset.org/annotations/{ANNOTATION_FILE}",f"{DATA_DIR}/")

print('\nextracting data from zip file')
# extract data
with ZipFile(f'{DATA_DIR}/{DATA_FILE}', 'r') as f:
    f.extractall(path=DATA_DIR)

with ZipFile(f'{DATA_DIR}/{ANNOTATION_FILE}', 'r') as f:
    f.extractall(path=DATA_DIR)

# removed zip files
print('Removing Zip files')
os.remove(f'{DATA_DIR}/{DATA_FILE}')
os.remove(f'{DATA_DIR}/{ANNOTATION_FILE}')