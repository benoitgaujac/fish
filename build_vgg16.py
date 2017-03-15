from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import MaxPool2DLayer as MaxPoolLayer
from lasagne.layers import TransformerLayer
#from lasagne.layers import Conv2DLayer as CPUConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as GPUConvLayer
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import elu
from lasagne.init import HeUniform
from lasagne.init import Constant

import numpy as np
import theano
import pdb

dropout_p = 0.5

def build_model(input_var,nclasses,GPU=False):
    if GPU:
        from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    else:
        from lasagne.layers import Conv2DLayer as Conv2DLayer

    l_in = InputLayer(shape=(None, 3, 224*5, 224*3), input_var=input_var)
    ################### ST ###################
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()
    loc_l1 = MaxPoolLayer(l_in, pool_size=(2, 2))
    loc_l2 = Conv2DLayer(
                loc_l1,
                num_filters=20,
                filter_size=(5, 5),
                W=HeUniform('relu'),
                nonlinearity=elu)
    loc_l3 = MaxPoolLayer(loc_l2, pool_size=(2, 2))
    loc_l4 = Conv2DLayer(
                loc_l3,
                num_filters=20,
                filter_size=(5, 5),
                W=HeUniform('relu'),
                nonlinearity=elu)
    loc_l5 = DenseLayer(
                loc_l4,
                num_units=50,
                W=HeUniform('relu'),
                nonlinearity=elu)
    loc_out = DenseLayer(
                loc_l5,
                num_units=6,
                b=b,
                W=Constant(0.0),
                nonlinearity=None)
    # Transformer network
    l_trans1 = TransformerLayer(l_in, loc_out, downsample_factor=(5.0,3))

    """
    from lasagne.layers import get_all_params
    params_train= get_all_params(l_trans1)
    print(params_train)
    params_= get_all_params(l_trans1,trainable=True)
    print(params_)
    from lasagne.layers import get_output_shape
    shpe = get_output_shape(l_trans1)
    pdb.set_trace()
    """

    ################### Conv1 ###################
    l_conv1_1 = Conv2DLayer(
                l_trans1,
                num_filters=64, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv1_2 = Conv2DLayer(
                l_conv1_1,
                num_filters=64, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    pool1 = PoolLayer(l_conv1_2, pool_size=(2, 2))
    ################### Conv2 ###################
    l_conv2_1 = Conv2DLayer(
                pool1,
                num_filters=128, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv2_2 = Conv2DLayer(
                l_conv2_1,
                num_filters=128, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    pool2 = PoolLayer(l_conv2_2, pool_size=(2, 2))
    ################### Conv3 ###################
    l_conv3_1 = Conv2DLayer(
                pool2,
                num_filters=256, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv3_2 = Conv2DLayer(
                l_conv3_1,
                num_filters=256, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv3_3 = Conv2DLayer(
                l_conv3_2,
                num_filters=256, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    pool3 = PoolLayer(l_conv3_3, pool_size=(2, 2))
    ################### Conv4 ###################
    l_conv4_1 = Conv2DLayer(
                pool3,
                num_filters=512, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv4_2 = Conv2DLayer(
                l_conv4_1,
                num_filters=512, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv4_3 = Conv2DLayer(
                l_conv4_2,
                num_filters=512, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    pool4 = PoolLayer(l_conv4_3, pool_size=(2, 2))
    ################### Conv5 ###################
    l_conv5_1 = Conv2DLayer(
                pool4,
                num_filters=512, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv5_2 = Conv2DLayer(
                l_conv5_1,
                num_filters=512, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    l_conv5_3 = Conv2DLayer(
                l_conv5_2,
                num_filters=512, filter_size=(3,3),
                stride=(1, 1), pad=1,
                nonlinearity=elu,
                flip_filters=False)
    pool5 = PoolLayer(l_conv5_3, pool_size=(2, 2))
    ################### fc6 ###################
    fc6 = DenseLayer(
    		pool5,
            nonlinearity=elu,
            num_units=4096)
    fc6_dropout = DropoutLayer(fc6, p=dropout_p)
    ################### fc7 ###################
    fc7 = DenseLayer(
    		fc6_dropout,
            nonlinearity=elu,
            num_units=4096)
    fc7_dropout = DropoutLayer(fc7, p=dropout_p)
    ################### Classification ###################
    fc8 = DenseLayer(
    		fc7_dropout,
            num_units=nclasses,
            nonlinearity=None)
    network = NonlinearityLayer(fc8, softmax)

    return network
