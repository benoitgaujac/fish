import os
import sys
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

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-d', '--im_dir', action='store', dest='im_dir',
    help="images directory")


def move_and_rename_data(file_dir):
    root_dir = "../fish_results/train_data"
    dst_dir = os.path.join(root_dir,os.path.basename(file_dir))
    c = 0
    for dir_name, _, file_list in os.walk(file_dir):
        for file_name in file_list:
            image_name = os.path.join(dir_name, file_name)
            im = image.load_img(image_name, target_size=None)
            dst_name = "benoit_" + file_name[:-5] + ".jpg"
            im.save(os.path.join(dst_dir, dst_name))
            c+=1
            if (c+1)%200==0:
                print("{} images processeded".format(c+1))

def load_data(dir_data):
    images = []
    c=0
    for root_dir, sub_dir_list, file_list in os.walk(dir_data):
        for sub_dir in sub_dir_list:
            sub_dir_path = os.path.join(root_dir,sub_dir)
            for dir_name, _, file_list in os.walk(sub_dir_path):
                for file_name in file_list:
                    file_path = os.path.join(dir_name,file_name)
                    image = load_image(file_path)
                    images.append(image)
                    c+=1
                    if c%500==0:
                        print("{} images loaded".format(c))
    return images

def load_image(imge_path):
    img = image.load_img(imge_path, target_size = (width, height))
    arr = image.img_to_array(img,data_format='channels_first').astype('float32')
    return arr

def get_mean(list_im):
    images = np.stack([image.img_to_array(im,data_format='channels_first') for im in list_im]).astype('float32')
    pdb.set_trace()
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
    all_img = []
    c=0
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
                    images.append([im.copy(),file_name])
                    all_img.append(im.copy())
                    im.close()
                    label = FISH_CLASSES[j]
                    labels.append(label)
                else:
                    print("Unsupported extension: {}".format(os.path.join(dir_name, file_name)))
            if (c+1)%500==0:
                print("{} images loaded".format(c+1))
            c+=1
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
    options, arguments = parser.parse_args(sys.argv)
    images=load_data(options.im_dir)
    pdb.set_trace()
    get_mean(images)
    #split_dataset("train", "dataset", 0.15)
    #move_and_rename_data(options.im_dir)


    #dataset = test_dataset("/Users/benoitgaujac/Documents/UCL/Applied ML/kaggle/Fisheries/test_stg1", 64)
    #mean = get_mean(dataset.images)
    #tr = dataset.train_set
    #vl = dataset.val_set
    #batches_tr = dataset.iterate_minibatches()
    #for batch in batches_tr:
    #    im, lab = batch

    #batches_vl = dataset.iterate_minibatches(vl)
    #split_dataset("train", "dataset", 0.15)
