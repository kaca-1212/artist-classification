#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

# fixed-sizes for image
fixed_size = tuple((256, 256))

# path to training data
path = "dataset/train"

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# feature-descriptor-1: Hu Moments
# def fd_hu_moments(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     feature = cv2.HuMoments(cv2.moments(image)).flatten()
#     return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# def fd_sift(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     sift = cv2.SIFT()
#     kp = sift.detect(gray,None)
#     return kp

def fd_hog(image):
    hog = cv2.HOGDescriptor()
    h = hog.compute(image)
    return h

# # get the training labels
# train_labels = os.listdir(train_path)

# # sort the training labels
# train_labels.sort()
# print(train_labels)

# empty lists to hold feature vectors and labels
def get_feature_vectors(path):
    global_features = []
    labels = []

    # # num of images per class
    # images_per_class = 80

    # loop over the training data sub-folders
    for file in os.listdir(path):
        
        # get the current training label
        ### current_label = training_name
        ### DO LATER

       

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hog = fd_hog(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hog])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)


    print "[STATUS] completed Global Feature Extraction..."

    # get the overall feature vector size
    print "[STATUS] feature vector size {}".format(np.array(global_features).shape)

    # get the overall training label size
    print "[STATUS] training Labels {}".format(np.array(labels).shape)

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print "[STATUS] training labels encoded..."

    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print "[STATUS] feature vector normalized..."

    print "[STATUS] target labels: {}".format(target)
    print "[STATUS] target labels shape: {}".format(target.shape)

    # save the feature vector using HDF5
    h5f_data = h5py.File('output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File('output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print "[STATUS] end of training.."

    return rescaled_features