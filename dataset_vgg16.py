import os
import random
import json
import pdb

import operator
import math
import numpy as np

from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
max_size = 1024
SEED = 66478

def get_mean(list_im):
    images = np.stack([im[0] for im in list_im]).astype('float32')
    mean = np.mean(images,(2,3))
    mean = np.mean(np.transpose(mean,(1,0)),(1))
    print("Mean on training set: {}, {}, {}".format(mean[0],mean[1],mean[2]))
    np.savez("training_mean.npz", mean)

class Dataset:
    def __init__(self, root_dir, image_size, batch_size, val_size):
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_size = max_size
        self.FISH_CLASSES = FISH_CLASSES
        self.init_images(root_dir)
        self.split_dataset(val_size)

    def init_images(self, root_dir):
        self.images = []
        mean = np.load("training_mean.npz")
        mean = mean[mean.keys()[0]]
        Mmean = np.ones((3,self.image_size,self.image_size))
        for i in range(3):
            Mmean[i] *= mean[i]
        #for i in range(len(FISH_CLASSES)):
        for i in range(2,3):
            c = 0
            im_dir = os.path.join(root_dir, FISH_CLASSES[i])
            for dir_name, _, file_list in os.walk(im_dir):
                for file_name in file_list:
                    if file_name.endswith('.png') or file_name.endswith('.jpg'):
                        image_file = os.path.join(dir_name, file_name)
                        image = self.preprocess_image(image_file)
                        image -= Mmean
                        if image is None:
                            print("Error loading image: {}".format(image_file))
                            continue
                        label = i
                        self.images.append((image, label, file_name))
                    else:
                        print("Unsupported extension: {}".format(os.path.join(dir_name, file_name)))
                    c+=1
            print("{} done: {} images.".format(FISH_CLASSES[i],c))
        #self.print_stats()
        print("Total images: {}".format(len(self.images)))

    def preprocess_image(self,filename):
        img = image.load_img(filename, target_size = (self.image_size, self.image_size))
        arr = image.img_to_array(img)
        return arr
        #return np.transpose(arr,(2,0,1))
    """
    def get_image(self, filename):
        image = skimage.io.imread(filename)
        if image is None or len(image.shape) < 2 or len(image.shape) > 3:
           print("Fatal error can't read image: {}".format(image_file))
           return None, None
        image = skimage.transform.resize(image, (self.image_size, self.image_size), preserve_range=True)
        image = np.transpose(image,(2,0,1))
        return image
    """

    def print_stats(self):
        stats = dict()
        for species in self.FISH_CLASSES:
            stats[species] = 0
        for image in self.images:
            lab = image[1]
            stats[self.FISH_CLASSES[lab]] += 1
        #print("Dataset stats:")
        print("Total images: {}".format(len(self.images)))
        labels = sorted(stats.items(), key=operator.itemgetter(1))
        for label in labels:
            print("{}: {}".format(label[0], label[1]))
        print("")

    def split_dataset(self,val_size):
        if val_size!=1:
            X_all = [im[0] for im in self.images]
            y_all = [[im[1]] for im in self.images]
            X_train, X_valid, y_train, y_valid = train_test_split(  X_all, y_all,
                                                                    test_size=val_size,
                                                                    random_state=SEED,
                                                                    stratify=y_all)
            self.train_set = [(X_train[i],y_train[i]) for i in range(len(y_train))]
            self.val_set = [(X_valid[i],y_valid[i]) for i in range(len(y_valid))]
        else:
            self.train_set = []
            self.val_set = self.images

    def iterate_minibatches(self, images, testing=False):
        batches = []
        X_ = np.stack([im[0] for im in images]).astype('float32')
        y_ = np.stack([im[1] for im in images]).astype('float32')
        y_ = np.reshape(y_,[-1])
        if not testing:
            X, y = shuffle(X_, y_)
        for i in range(int(len(X)/self.batch_size)+1):
            X_batch = X[i * self.batch_size: (i + 1) * self.batch_size]
            y_batch = y[i * self.batch_size: (i + 1) * self.batch_size]
            X_id = []
            if testing:
                X_id = images[i * self.batch_size: (i + 1) * self.batch_size][2]
            batches.append([X_batch, y_batch, X_id])
        return batches

if __name__ == '__main__':
    dataset = Dataset("/Users/benoitgaujac/Documents/UCL/Applied ML/kaggle/Fisheries/train", 224, 32, 0.2)
    mean = get_mean(dataset.images)
    #tr = dataset.train_set
    #vl = dataset.val_set
    #batches_tr = dataset.iterate_minibatches(tr)
    #pdb.set_trace()
    #batches_vl = dataset.iterate_minibatches(vl)
