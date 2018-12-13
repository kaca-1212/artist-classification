from os.path import join

import h5py

import time


import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.regularizers import l2

MAX_EPOCHS = 500
BATCH_SIZE = 96
L2_REG = 0.003
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
SOFTMAX_SIZE = 15
data_path = 'dataset/cnn_data'
batch_size = 16

def _cnn(imgs_dim, compile_=True):
    model = Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=imgs_dim))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    """
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
     

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    """

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    """
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU()
     
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    """

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    """
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
        
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    """

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2,2)))

    """
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
        
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    """
    
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))
    
    model.add(Flatten())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization())
    model.add(ReLU())

    if compile_:
        model.add(Dropout(rate=0.5))
        model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
        model.add(BatchNormalization())
        model.add(Activation(activation='softmax'))
        return compile_model(model)

    return model

    
def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(
        filters=nb_filter, kernel_size=(3,3), input_shape=input_shape,
        border_mode='same', kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        filters=nb_filter, kernel_size=(3,3), padding='same',
        kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))

def _big_convolutional_layer(nb_filter):
    return Conv2D(
        filters=nb_filter, kernel_size=(5,5), 
        border_mode='same', kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))

def _dense_layer(output_dim):
    return Dense(output_dim=output_dim, kernel_regularizer=l2(l=L2_REG), kernel_initializer=W_INIT)


def compile_model(model):
    adam = Adam(lr=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    return model


def load_trained_cnn_feature_maps_layer(model_path):
    return _load_trained_cnn_layer(model_path, LAST_FEATURE_MAPS_LAYER)


def load_trained_cnn_penultimate_layer(model_path):
    return _load_trained_cnn_layer(model_path, PENULTIMATE_LAYER)


def load_trained_cnn_softmax_layer(model_path):
    return _load_trained_cnn_layer(model_path, SOFTMAX_LAYER)


def _load_trained_cnn_layer(model_path, layer_index):
    model = load_model(model_path)
    dense_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_index].output])
    # output in test mode = 0
    return lambda X: dense_output([X, 0])[0]

IMGS_DIM_3D = (256, 256,3)
def _run_model():
    model = _cnn(IMGS_DIM_3D)
    predict_datagen = ImageDataGenerator()

    predict_generator = predict_datagen.flow_from_directory(data_path + '/test', target_size=(256,256),batch_size = batch_size, class_mode = 'categorical')

    model.load_weights('run4.h5')
    start = time.time()
    print(model.metrics_names)
    print(model.evaluate_generator(predict_generator,steps=47))
    end = time.time()-start
    print(end/743)

if __name__ == '__main__':
    _run_model()
