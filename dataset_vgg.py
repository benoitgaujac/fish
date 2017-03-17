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


class train_dataset:
    def __init__(self, root_dir, batch_size):
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.FISH_CLASSES = FISH_CLASSES
        self.init_images(root_dir)

    def init_images(self, root_dir):
        self.images = []
        self.labels = []
        # Substract training mean
        mean = np.load("training_mean.npz")
        mean = mean[mean.keys()[0]]
        """
        Mmean = np.ones((3,self.width,self.height))
        for i in range(3):
            Mmean[i] *= mean[i]
        """
        #for i in range(len(FISH_CLASSES)):
        for i in range(1,3):
            c = 0
            im_dir = os.path.join(root_dir, FISH_CLASSES[i])
            for dir_name, _, file_list in os.walk(im_dir):
                for file_name in file_list:
                    if file_name.endswith('.png') or file_name.endswith('.jpg'):
                        image_file = os.path.join(dir_name, file_name)
                        image = self.read_and_process_image(image_file)
                        #image -= Mmean
                        image -= mean.reshape((3,1,1))
                        if image is None:
                            print("Error loading image: {}".format(image_file))
                            continue
                        label = i
                        self.images.append(image)
                        self.labels.append(label)
                    else:
                        print("Unsupported extension: {}".format(os.path.join(dir_name, file_name)))
                    c+=1
            print("{} done: {} images.".format(FISH_CLASSES[i],c))
        #self.images = np.stack([im_ for im_ in im]).astype('float32')
        #self.labels = np.stack([lab_ for lab_ in lab]).astype('float32')
        print("Total {} images: {}\n".format(root_dir[-8:],len(self.images)))

    def read_and_process_image(self,filename):
        img = image.load_img(filename, target_size = (self.width, self.height))
        arr = image.img_to_array(img,data_format='channels_first').astype('float32')
        return arr
        #return np.transpose(arr,(2,0,1))

    def iterate_minibatches(self, shuffle=False):
        assert len(self.images) == len(self.labels)
        if shuffle:
            indices = np.arange(len(self.images))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.images) - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
                im_batch = [self.images[i] for i in excerpt]
                lab_bacth = [self.labels[i] for i in excerpt]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
                im_batch = self.images[excerpt]
                lab_bacth = self.labels[excerpt]
            yield im_batch, lab_bacth

class test_dataset:
    def __init__(self, root_dir, batch_size):
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.init_images(root_dir)

    def init_images(self, root_dir):
        self.images = []
        self.images_id = []
        # Substract training mean
        mean = np.load("training_mean.npz")
        mean = mean[mean.keys()[0]]
        Mmean = np.ones((3,self.width,self.height))
        for i in range(3):
            Mmean[i] *= mean[i]
        for dir_name, _, file_list in os.walk(root_dir):
            for file_name in file_list:
                if file_name.endswith('.png') or file_name.endswith('.jpg'):
                    image_file = os.path.join(dir_name, file_name)
                    image = self.read_and_process_image(image_file)
                    image -= Mmean
                    if image is None:
                        print("Error loading image: {}".format(image_file))
                        continue
                    self.images.append(image)
                    self.images_id.append(file_name)
                else:
                    print("Unsupported extension: {}".format(os.path.join(dir_name, file_name)))
        print("Total test images: {}\n".format(len(self.images)))

    def read_and_process_image(self,filename):
        img = image.load_img(filename, target_size = (self.width, self.height))
        arr = image.img_to_array(img,data_format='channels_first').astype('float32')
        return arr

    def iterate_minibatches(self):
        assert len(self.images) == len(self.images_id)
        for start_idx in range(0, len(self.images) - self.batch_size + 1, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            im_batch = self.images[excerpt]
            lab_bacth = self.images_id[excerpt]
            yield im_batch, lab_bacth


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
