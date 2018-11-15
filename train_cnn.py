
# from https://github.com/inejc/painters/blob/master/painters/train_cnn.py

from os.path import join

import h5py

import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.layers.advanced_activations import ReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.regularizers import l2


IMGS_DIM_3D = (256, 256,3)
MAX_EPOCHS = 500
BATCH_SIZE = 96
L2_REG = 0.003
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
SOFTMAX_SIZE = 29
data_path = 'dataset/cnn_data'
batch_size = 16

def load_img_arr(p):
    return img_to_array(load_img(p))

def make_fit_gen(path):
    train_datagen = ImageDataGenerator( featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=180,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
fill_mode='reflect')
    
    val_datagen = ImageDataGenerator(featurewise_center = True, featurewise_std_normalization=True)

    train_generator = train_datagen.flow_from_directory(data_path + '/train', target_size=(256,256),batch_size = batch_size, class_mode = 'categorical')
    val_generator = val_datagen.flow_from_directory(data_path + '/validation', target_size=(256,256),batch_size = batch_size, class_mode = 'categorical')
    
    return (train_generator, val_generator)

def preprocess_data(path):
    img_lst = []
    label_lst = []

    for file in os.listdir(path):
        artist = file.split('_')[0]
        img_vec = img_to_array(load_img(file))
        img_lst.append(img_vec)
        label_lst.append(np.str(artist))

    # save the feature vector using HDF5
    h5f_data = h5py.File('output/cnn_data.h5', 'w+')
    h5f_data.create_dataset('cnn_dataset_1', data=np.array(img_lst))

    h5f_label = h5py.File('output/cnn_labels.h5', 'w+')
    h5f_label.create_dataset('cnn_dataset_1', data=label_lst)

    h5f_data.close()
    h5f_label.close()


def _train_model():
    """
    h5f_data = h5py.File('output/cnn_data.h5', 'r')
    h5f_label = h5py.File('output/cnn_labels.h5', 'r')

    training_data_str = h5f_data['cnn_dataset_1']
    training_labels_str = h5f_label['cnn_dataset_1']

    training_data = list(np.array(training_data_str))
    training_labels = np.array(training_labels_str)

    h5f_data.close()
    h5f_label.close()
    """

    model = _cnn(IMGS_DIM_3D)
    model = compile_model(model)
    gens = make_fit_gen('dataset/cnn_data')
    model.fit_generator(gens[0], epochs = 200, validation_data = gens[1])
    model.save_weights('run1.h5')

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
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=-1))
    model.add(ReLU())
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
    adam = Adam(lr=0.000074)
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


if __name__ == '__main__':
    _train_model()
