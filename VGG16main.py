import sys
import os
import time

import numpy as np
import pickle
import theano
import theano.tensor as T
import lasagne
from six.moves import urllib

import dataset_vgg16
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
parser.add_option('-w', '--weight_dir', action='store', dest='weight_dir',
    help="training from VGG weights or from own pretrained weights [vgg,nost,st]")

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
        params[8:-2] =  weights[:-2]
    elif weight_dir=="nost":
        data = np.load(weight_path)
        pretrained_weights = data[data.keys()[0]]
        params[8:] = pretrained_weights
    elif weight_dir=="st":
        data = np.load(weight_path)
        pretrained_weights = data[data.keys()[0]]
        params = pretrained_weights

    #network.initialize_layers()
    lasagne.layers.set_all_param_values(network, params)
    return network

######################################## main ########################################
def main(weight_dir, GPU=False, num_epochs=50) :
    # load training data
    print("loading training data...")
    train_data_set = dataset_vgg16.train_dataset("/Users/benoitgaujac/Documents/UCL/Applied ML/kaggle/Fisheries/dataset/training", batch_size)
    val_data_set = dataset_vgg16.train_dataset("/Users/benoitgaujac/Documents/UCL/Applied ML/kaggle/Fisheries/dataset/validati", batch_size)
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
    params_st = params[:8]
    params_fc = params[-2:]
    params_to_train = params_st + params_fc
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

################################################################################
if __name__ == '__main__':
    options, arguments = parser.parse_args(sys.argv)
    # mode test or train
    main(options.weight_dir,options.ressources=="GPU")
