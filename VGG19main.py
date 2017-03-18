import sys
import os
import time

import numpy as np
import pickle
import theano
import theano.tensor as T
from keras.preprocessing.image import ImageDataGenerator
import lasagne
from six.moves import urllib
import pandas as pd

import dataset_vgg
import build_vgg19_v2 as build_vgg19

import pdb
import csv

TEST_DIR = "test_stg1/"
WEIGHTS_DIR = "pretrained_weights"
SOURCE_URL = "https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/"
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
NUM_CLASSES = 8
CHANNELS = 3
IM_SIZE_vgg = 224
batch_size = 64
test_batch_size = 200
nbr_augmentation = 5

from optparse import OptionParser
parser = OptionParser()
parser.add_option('-r', '--ressources', action='store', dest='ressources',
    help="ressources to compute [CPU, GPU]")
parser.add_option('-w', '--weights', action='store', dest='weights',
    help="training from VGG weights or from own pretrained weights [vgg,fish]")
parser.add_option('-d', '--data_dir', action='store', dest='data_dir',
    help="root_dir. Should contain dataset folder with training and validati folders, and test_stg1 folder")
parser.add_option('-m', '--mode', action='store', dest='mode',
    help="mode: training or testing")

######################################## Utils ########################################
def maybe_download(filename):
    if not os.path.isdir(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)
    filepath = os.path.join(WEIGHTS_DIR, filename)
    if not os.path.isfile(filepath):
        print("Downloading pre-trained weights from s3...")
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        #filepath, _ = urllib.request.urlretrieve("https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl", filepath)
        print('Successfully downloaded vgg19 pre trained weights')

def load_params(network,weight_path,weight_dir):
    params = lasagne.layers.get_all_param_values(network)
    if weight_dir=="vgg":
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f, encoding='latin-1')['param values']
        params[:-6] =  weights[:-3]
    elif weight_dir=="fish":
        data = np.load(weight_path)
        pretrained_weights = data[data.keys()[0]]
        params = pretrained_weights
    lasagne.layers.set_all_param_values(network, params)
    return network

######################################## main ########################################
def main(datat_dir, weight_dir, GPU=False, training=False, num_epochs=25, data_augmentation=True):
    if training:
        # load training data
        print("\nloading training data...")
        train_dir = os.path.join(datat_dir, "dataset/training")
        train_data_set = dataset_vgg.train_dataset(train_dir, batch_size)
        val_dir = os.path.join(datat_dir, "dataset/validati")
        val_data_set = dataset_vgg.train_dataset(val_dir, batch_size)

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        # Downloading weights if needed
        if weight_dir=="vgg":
            weight_path = os.path.join(WEIGHTS_DIR, "vgg19.pkl")
            maybe_download("vgg19.pkl")
        else:
            weight_path = os.path.join(WEIGHTS_DIR, "model_vgg19.npz")
        # Build CNN model
        print("Building network and compiling functions...")
        network = build_vgg19.build_model(input_var,NUM_CLASSES,GPU)
        network = load_params(network,weight_path,weight_dir)
        # Create a loss expression for training (categorical crossentropy)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # Create update expressions for training (Stochastic Gradient Descent with Nesterov momentum)
        learning_rate = 0.0007
        # Create update expression
        params = lasagne.layers.get_all_params(network, trainable=True)
        params_fc = params[-5:]
        params_to_train = params_fc
        #updates = lasagne.updates.nesterov_momentum(loss, params[-2:], learning_rate=learning_rate, momentum=0.9)
        updates = lasagne.updates.adam(loss, params_to_train, learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
        # Create a loss expression for validation/testing
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        # Create an expression for the classification accuracy
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
        # Compile a function performing a training step on a mini-batch (by giving
        #the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)
        # Opening csv file
        csvfileTrain = open('Perf/vgg19.csv', 'w')
        Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
        Trainwriter.writerow(['Num Epoch', 'Time', 'Training loss', 'Validation loss','Validation accuracy','best accuracy','best val'])

        ######################################## training ########################################
        print("\nStarting training ...")
        #best val
        best_val = 100.0
        best_acc = 0.0
        epoch = 0
        epochs_without_improvement = 0
        max_no_improvement = 3
        # We iterate over epochs:
        while epoch<num_epochs:
            if (epochs_without_improvement+1)%3==0 :
                learning_rate =float(learning_rate)/2
                updates = lasagne.updates.adam(loss, params_to_train, learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            # In each epoch, do a full pass over the training data:
            print("Start training original data..")
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in train_data_set.iterate_minibatches(shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1
            # If data augmentation
            print("Training original data done")
            if data_augmentation:
                print("Start training augmented data..")
                datagen = ImageDataGenerator(
                                featurewise_center=False,  # set input mean to 0 over the dataset
                                samplewise_center=False,  # set each sample mean to 0
                                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                samplewise_std_normalization=False,  # divide each input by its std
                                zca_whitening=False,  # apply ZCA whitening
                                rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                horizontal_flip=True,  # randomly flip images
                                vertical_flip=False, # randomly flip images
                                shear_range=0.2, # randomly shear images, shear Intensity
                                zoom_range=0.2, # randomly zoom images: [lower, upper] = [1-zoom_range, 1+zoom_range]
                                fill_mode='nearest', # fills points outside the boundaries of the input woth nearest
                                data_format="channels_first") # format input images: (samples, channels, height, width)
                random_seed = np.random.random_integers(0, 100000)
                train_imgen = np.stack(train_data_set.images,axis=0)
                train_labgen = np.stack(train_data_set.labels,axis=0)
                train_batch_generator = datagen.flow(
                                            train_imgen, train_labgen,
                                            batch_size=batch_size,
                                            shuffle = True,
                                            seed = random_seed)
                nb_batch_train = len(train_data_set.images)/batch_size
                c=0
                for batch in train_batch_generator:
                    inputs, targets = batch
                    train_err += train_fn(inputs, targets)
                    train_batches += 1
                    c+=1
                    if c>=nb_batch_train:
                        break
                print("Training augmented data done")

            # And a full pass over the validation data:
            val_err = 0
            val_batches = 0
            val_acc = 0
            for batch in val_data_set.iterate_minibatches(shuffle=True):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            # if data augmenetation
            if data_augmentation:
                random_seed = np.random.random_integers(0, 100000)
                val_imgen = np.stack(val_data_set.images,axis=0)
                val_labgen = np.stack(val_data_set.labels,axis=0)
                val_batch_generator = datagen.flow(
                                            val_imgen, val_labgen,
                                            batch_size=batch_size,
                                            shuffle = True,
                                            seed = random_seed)
                nb_batch_val = len(val_data_set.images) / batch_size
                c=0
                for batch in val_batch_generator:
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    val_err += err
                    val_acc += acc
                    val_batches += 1
                    c+=1
                    if c>=nb_batch_val:
                        break

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("Learning rate: {:.6f}".format(learning_rate))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.3f} %".format(val_acc / val_batches * 100))
            epoch += 1
            if ((val_err / val_batches) <= best_val) :
                best_val = val_err / val_batches
                best_acc = val_acc / val_batches * 100
                epochs_without_improvement = 0
                # save network
                weight_path = os.path.join(WEIGHTS_DIR, "model_vgg19.npz")
                np.savez(weight_path, lasagne.layers.get_all_param_values(network))
                print("MODEL SAVED")
            else:
                epochs_without_improvement += 1
            print("lowest val {:.8f}, best acc = {:1.2f}, epochs without improvement = {}\n".format(best_val,best_acc,epochs_without_improvement))
            # Writing csv file with results
            Trainwriter.writerow([epoch, time.time() - start_time,
                                (train_err / train_batches),
                                (val_err / val_batches),
                                (val_acc / val_batches),
                                best_acc, best_val])

    ######################################## testing ########################################
    print("\nLoading test data...")
    test_dir = os.path.join(datat_dir,"test_stg1")
    test_data_set = dataset_vgg.test_dataset(test_dir, test_batch_size)
    if not training:
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        # Downloading weights if needed
        weight_path = os.path.join(WEIGHTS_DIR, "model_vgg19.npz")
        #weight_path = os.path.join(WEIGHTS_DIR, "vgg19.pkl")
        #maybe_download("vgg19.pkl")
        # Build CNN model
        print("Building network and compiling functions...")
        network = build_vgg19.build_model(input_var,NUM_CLASSES,GPU)
        network = load_params(network,weight_path,weight_dir)
        # Create a loss expression for validation/testing
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        print("model build..")
    # Compile a second function computing the validation loss and accuracy:
    test_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('Testing original test set ...')
    start_time = time.time()
    predictions = []
    prediction = []
    im_id = []
    for batch in test_data_set.iterate_minibatches():
        inputs, inputs_id = batch
        pred = test_fn(inputs)
        im_id += inputs_id
        prediction.append(pred)
    predictions.append(np.concatenate(prediction, axis=0))
    print("Original set done, took {:.2f}s\n".format(time.time()-start_time))
    # Data augmentation
    test_datagen = ImageDataGenerator(
                    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    shear_range=0.1, # randomly shear images, shear Intensity
                    zoom_range=0.1, # randomly zoom images: [lower, upper] = [1-zoom_range, 1+zoom_range]
                    fill_mode='nearest', # fills points outside the boundaries of the input woth nearest
                    data_format="channels_first") # format input images: (samples, channels, height, width)
    test_imgen = np.stack(test_data_set.images,axis=0)
    for idx in range(nbr_augmentation):
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('{}th augmentation for testing ...'.format(idx+1))
        start_time = time.time()
        random_seed = np.random.random_integers(0, 100000)
        test_generator = test_datagen.flow(test_imgen,batch_size=test_batch_size,shuffle=False,seed=random_seed) # make sure shuffle = False !!!
        prediction = []
        nb_batch = len(test_data_set.images)/test_batch_size
        c=0
        for batch in test_generator:
            inputs = batch
            pred = test_fn(inputs)
            prediction.append(pred)
            c+=1
            print("Batch {} done".format(c))
            if c>=nb_batch:
                break
        predictions.append(np.concatenate(prediction, axis=0))
        print("{}th augmentation set done, took {:.2f}s\n".format(idx+1,time.time()-start_time))
    # Stack prediction to average
    stack_preds = np.stack(predictions, axis=0)
    # Sanity check
    if len(np.shape(stack_preds))!=3 or np.shape(stack_preds)[0]!=(1+nbr_augmentation):
        raise Exception("Error in test predictions")
    # Average predictions
    average_preds = np.average(stack_preds,axis=0)
    # Sanity check
    if len(np.shape(average_preds))!=2 or np.shape(average_preds)[0]!=(len(im_id)):
        raise Exception("Error in average predictions")
    submission = pd.DataFrame(average_preds, columns=FISH_CLASSES)
    submission.insert(0, 'image', im_id)
    submission.to_csv('./submission.csv', index=False)
    print("Submission file successfully created.")


################################################################################
if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    # mode test or train
    main(options.data_dir,options.weights,options.ressources=="GPU",options.mode=="training",data_augmentation=False)
