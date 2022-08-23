import pandas as pd
from sklearn.model_selection import train_test_split

import model
from generator import SegmentationGenerator

def create_generators(imgs_path, labels_path, test_size=0.3):
    labels_df = pd.read_csv(labels_path)
    labels_df["HasShip"] = labels_df["EncodedPixels"].notnull().astype(int)
    unique_imgs = labels_df.groupby("ImageId").sum().rename({"HasShip": "TotalShips"}, axis=1)

    train_ids, valid_ids = train_test_split(unique_imgs,
                    test_size = test_size,
                    stratify = unique_imgs['TotalShips'])
    train_df = pd.merge(labels_df, train_ids, left_on="ImageId", right_on="ImageId")
    valid_df = pd.merge(labels_df, valid_ids, left_on="ImageId", right_on="ImageId")

    segm_train_df = train_df[train_df["HasShip"] != 0].drop_duplicates("ImageId")
    segm_valid_df = valid_df.drop_duplicates("ImageId")

    segm_training_generator = SegmentationGenerator(segm_train_df["ImageId"],
                                segm_train_df, imgs_path, batch_size=8)
    segm_validation_generator = SegmentationGenerator(segm_valid_df["ImageId"],
                                segm_valid_df, imgs_path, batch_size=8)