import os
import random
import json
import pdb
import shutil

import operator
import math
import numpy as np

from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
width = 224
height = 224
SEED = 66478

def get_mean(list_im):
    images = np.stack([im[0] for im in list_im]).astype('float32')
    mean = np.mean(images,(1,3))
    mean = np.mean(np.transpose(mean,(1,0)),(1))
    print("Mean on training set: {}, {}, {}".format(mean[0],mean[1],mean[2]))
    np.savez("training_mean.npz", mean)

def split_dataset(root_dir, dst_dir, val_size):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    else:
        # Remove old split
        shutil.rmtree(dst_dir)
        os.mkdir(dst_dir)
    for j in range(len(FISH_CLASSES)):
        images = []
        labels = []
        im_dir = os.path.join(root_dir, FISH_CLASSES[j])
        for dir_name, _, file_list in os.walk(im_dir):
            for file_name in file_list:
                if file_name.endswith('.png') or file_name.endswith('.jpg'):
                    image_file = os.path.join(dir_name, file_name)
                    im = image.load_img(image_file, target_size=None)
                    if im is None:
                        print("Error loading image: {}".format(image_file))
                        continue
                    label = FISH_CLASSES[j]
                    images.append([im,file_name])
                    labels.append(label)
                else:
                    print("Unsupported extension: {}".format(os.path.join(dir_name, file_name)))
        print("{}: read {} images.".format(FISH_CLASSES[j],len(images)))
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=val_size, random_state=SEED, stratify=labels)
        for i in range(len(X_train)):
            im = X_train[i][0]
            fname = X_train[i][1]
            label = y_train[i]
            dst_train = os.path.join(dst_dir,"training")
            if not os.path.isdir(dst_train):
                os.mkdir(dst_train)
            dst_path = os.path.join(dst_train,label)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            im.save(os.path.join(dst_path, fname))

        for i in range(len(X_val)):
            im = X_val[i][0]
            fname = X_val[i][1]
            label = y_val[i]
            dst_val = os.path.join(dst_dir,"validati")
            if not os.path.isdir(dst_val):
                os.mkdir(dst_val)
            dst_path = os.path.join(dst_val,label)
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            im.save(os.path.join(dst_path, fname))

        print("{} done:".format(FISH_CLASSES[j]))
        print("  train:\t{} images".format(len(X_train)))
        print("  valditaion:\t{} images\n".format(len(X_val)))

if __name__ == '__main__':
    dataset = test_dataset("/Users/benoitgaujac/Documents/UCL/Applied ML/kaggle/Fisheries/test_stg1", 64)
    #mean = get_mean(dataset.images)
    #tr = dataset.train_set
    #vl = dataset.val_set
    batches_tr = dataset.iterate_minibatches()
    for batch in batches_tr:
        im, lab = batch

    #batches_vl = dataset.iterate_minibatches(vl)
    #split_dataset("train", "dataset", 0.15)
