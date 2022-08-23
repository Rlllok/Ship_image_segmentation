import tensorflow as tf
import numpy as np
import cv2
import os


def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T
    

def get_multimask(rle_code_list, img_shape=(768,768)):
    multimask = np.zeros(img_shape)
    for rle in rle_code_list:
        if isinstance(rle, str):
            multimask += rle_decode(rle)
    return multimask


class SegmentationGenerator(tf.keras.utils.Sequence):
    def __init__(self, unique_ids, labels, imgs_path, batch_size):
        self.unique_ids = unique_ids
        self.labels = labels
        self.batch_size = batch_size
        self.imgs_path = imgs_path
        
    def __len__(self):
        return int(np.floor(len(self.unique_ids) / self.batch_size))
    
    def __getitem__(self, index):
        batch_ids = self.unique_ids[index * self.batch_size:(index+1) * self.batch_size]
        batch_X = np.array([cv2.imread(os.path.join(self.imgs_path, x)) for x in batch_ids])
        batch_Y = np.array([np.expand_dims(np.stack(get_multimask(self.labels[self.labels["ImageId"]==x]['EncodedPixels'].values), 0), -1)
                            for x in batch_ids])
        return batch_X, batch_Y