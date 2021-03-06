import sys
import os
import time

import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne
from six.moves import urllib
import pandas as pd

import dataset_vgg
import build_vgg16

import pdb
import csv

TEST_DIR = "test_stg1/"
WEIGHTS_DIR = "pretrained_weights"
SOURCE_URL = "https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/"
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
NUM_CLASSES = 8
CHANNELS = 3
IM_SIZE_vgg = 224
IM_SIZE_incv3 = 299
batch_size = 256

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
        print('Successfully downloaded vgg16 pre trained weights')

def load_params(network,weight_path,weight_dir):
    params = lasagne.layers.get_all_param_values(network)
    if weight_dir=="vgg":
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f, encoding='latin-1')['param values']
        params[:-2] =  weights[:-2]
    elif weight_dir=="fish":
        data = np.load(weight_path)
        pretrained_weights = data[data.keys()[0]]
        params = pretrained_weights

    lasagne.layers.set_all_param_values(network, params)
    return network

######################################## main ########################################
def main(datat_dir, weight_dir, GPU=False, training=False, num_epochs=50):
    if training:
        # load training data
        print("\nloading data...")
        train_dir = os.path.join(datat_dir, "dataset/training")
        train_data_set = dataset_vgg.train_dataset(train_dir, batch_size)
        val_dir = os.path.join(datat_dir, "dataset/validati")
        val_data_set = dataset_vgg.train_dataset(val_dir, batch_size)

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        # Downloading weights if needed
        if weight_dir=="vgg":
            weight_path = os.path.join(WEIGHTS_DIR, "vgg16.pkl")
            maybe_download("vgg16.pkl")
        else:
            weight_path = os.path.join(WEIGHTS_DIR, "model_vgg16.npz")
        # Build CNN model
        print("Building network and compiling functions...")
        network = build_vgg16.build_model(input_var,NUM_CLASSES,GPU)
        network = load_params(network,weight_path,weight_dir)
        # Create a loss expression for training (categorical crossentropy)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # Create update expressions for training (Stochastic Gradient Descent with Nesterov momentum)
        learning_rate = 0.001
        # Create update expression
        params = lasagne.layers.get_all_params(network, trainable=True)
        params_fc = params[-2:]
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
        csvfileTrain = open('Perf/vgg16.csv', 'w')
        Trainwriter = csv.writer(csvfileTrain, delimiter=';',)
        Trainwriter.writerow(['Num Epoch', 'Time', 'Training loss', 'Validation loss','Validation accuracy','best accuracy','best val'])

        # Launch the training loop:
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
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in train_data_set.iterate_minibatches(shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

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
                weight_path = os.path.join(WEIGHTS_DIR, "model_vgg16.npz")
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

    print("\nLoading test data...")
    test_dir = os.path.join(datat_dir,"test_stg1")
    test_data_set = dataset_vgg.test_dataset(test_dir, 100)
    if not training:
        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        # Downloading weights if needed
        weight_path = os.path.join(WEIGHTS_DIR, "model_vgg16.npz")
        # Build CNN model
        print("Building network and compiling functions...")
        network = build_vgg16.build_model(input_var,NUM_CLASSES,GPU)
        network = load_params(network,weight_path,weight_dir)
        # Create a loss expression for validation/testing
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        print("model build..")
    # Compile a second function computing the validation loss and accuracy:
    test_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)
    c=0
    im_id = []
    for batch in test_data_set.iterate_minibatches():
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        inputs, inputs_id = batch
        pred = test_fn(inputs)
        if c==0:
            submission = pd.DataFrame(pred, columns=FISH_CLASSES)
        else:
            pred_to_add = pd.DataFrame(pred, columns=FISH_CLASSES)
            submission = submission.append(pred_to_add,ignore_index=True)
        im_id += inputs_id
        c += 1
        print("Batch {} done, took {:.4f}s.".format(c,time.time()-start_time))
    submission.insert(0, 'image', im_id)
    submission.to_csv('./submission.csv', index=False)


################################################################################
if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    # mode test or train
    main(options.data_dir,options.weights,options.ressources=="GPU",options.mode=="training")
