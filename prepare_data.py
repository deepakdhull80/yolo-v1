import os
import json

import wget
from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split

import config

DATA_FILE = f"{config.dataset_name}.zip"
ANNOTATION_FILE = f"annotations_trainval{config.dataset_name[-4:]}.zip"
DATA_DIR = config.data_base_path

# download coco dataset
def downlaod_dataset():
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


### prepare train and val set for specific categories
def prepare_dataset():
    js = json.load(open(f'{DATA_DIR}/annotations/instances_val{config.dataset_name[-4:]}.json', 'r'))
    print('Keys', js.keys())
    cat_df = pd.DataFrame(js['categories'])
    image_df = pd.DataFrame(js['images'])
    anno_df = pd.DataFrame(js['annotations'])

    image_cols = ['file_name', 'id']
    anno_cols = ['image_id', 'bbox', 'category_id', 'id']
    image_df = image_df[image_cols]
    anno_df = anno_df[anno_cols]

    # join between to df
    data_df = pd.merge(anno_df, image_df.rename(columns={'id':'image_id'}), how='inner', on='image_id')
    print(f'[=INFO] dataframe size: {data_df.shape}')
    print(f"[=INFO] columns are: {data_df.columns}")

    # top 10 items we pick
    cat_li = list(data_df['category_id'].value_counts().iloc[:10].keys())
    tmp = cat_df[cat_df['id'].isin(cat_li)]
    ids = tmp['id']
    names = tmp['name']
    del tmp
    cat = dict(zip(ids,names))
    json.dump(cat, open(f'{DATA_DIR}/sampled_categories.json','w'))

    sdata_df = data_df[data_df['category_id'].isin(cat_li)]
    print(sdata_df.shape)

    ### balance category-1
    #TODO: remove less threshold area object first
    n_c1 = sdata_df[(sdata_df['category_id'] == 1)].shape[0]
    c1_index = sdata_df[sdata_df['category_id'] == 1].sample(int(n_c1*0.8)).index
    sdata_df.drop(index=c1_index, inplace=True)

    train, val = train_test_split(sdata_df, train_size=0.8, shuffle=True, random_state=1, stratify=sdata_df['category_id'])
    train.to_parquet(f"{DATA_DIR}/train.parquet",index=False)
    val.to_parquet(f"{DATA_DIR}/val.parquet",index=False)
    print("*** Completed ***")

if __name__ == '__main__':
    # downlaod_dataset()
    prepare_dataset()