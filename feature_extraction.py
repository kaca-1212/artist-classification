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
import sys
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
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

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

def fd_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    return kp

def fd_hog(image):
    hog = cv2.HOGDescriptor()
    img_squished = cv2.resize(image, (64,128))
    locations  = ((0,0),)
    h = hog.compute(image, locations = locations)
    return h.flatten()

# # get the training labels
# train_labels = os.listdir(train_path)

# # sort the training labels
# train_labels.sort()
# print(train_labels)

# empty lists to hold feature vectors and labels
def get_feature_vectors(path):
    global_features = []
    labels = []
    missed_files = []
    # # num of images per class
    # images_per_class = 80
    
    checkpoint = 1
    
    # loop over the training data sub-folders
    for ind, file in enumerate(os.listdir(path)):
        """
        if ind % 3000 == 2999:
            print sys.getsizeof(global_features)
            np.save('checkpoint' + str(checkpoint), np.array(global_features))
            np.save('labels'+str(checkpoint),np.array(labels))
            global_features = []
            labels = []
            checkpoint = checkpoint + 1
        """
        # get the current training label
        current_label = file.split('_')[0]
        ### DO LATER
        
        print file



        # read the image and resize it to a fixed-size
        try:
            image = cv2.imread(path+'/'+file)
        except:
            missed_files.append(file)
            continue
        if image is None:
            continue
        #print str(image.shape[0]*image.shape[1]*image.shape[2])
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        #fv_hog = fd_hog(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        fv_hu = fd_hu_moments(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

   # Save the last few
    """
    print sys.getsizeof(global_features)
    np.save('checkpoint' + str(checkpoint), np.array(global_features))
    np.save('labels'+str(checkpoint),np.array(labels))
    print np.array(global_features).shape
    global_features = []
    checkpoint = checkpoint + 1
    
    checkpoint = 6
    global_features = np.row_stack((np.load('checkpoint1.npy'), np.load('checkpoint2.npy'), np.load('checkpoint3.npy'), np.load('checkpoint4.npy'), np.load('checkpoint5.npy')  ))
    labels = []
    for c in range(1, checkpoint):
        #global_features.extend(list(np.load('reducedpoint' + str(c)+ '.npy')))
        labels.extend(list(np.load('labels' + str(c)+ '.npy')))
    """
    print "[STATUS] completed Global Feature Extraction..."

    # get the overall feature vector size
    #print "[STATUS] feature vector size {}".format(np.array(global_features).shape)

    # get the overall training label size
    #print "[STATUS] training Labels {}".format(np.array(labels).shape)

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print "[STATUS] training labels encoded..."

    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(global_features)
    #global_features = scaler.fit_transform(global_features)
    print "[STATUS] feature vector normalized..."

    print "[STATUS] target labels: {}".format(target)
    print "[STATUS] target labels shape: {}".format(target.shape)
    
    # save the feature vector using HDF5
    h5f_data = h5py.File('output/data3.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(global_features))

    h5f_label = h5py.File('output/labels3.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

    print "[STATUS] end of training.."

    return global_features