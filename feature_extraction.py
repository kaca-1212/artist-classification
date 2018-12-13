#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------

# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
import gist
import math
import numpy as np
import mahotas
import cv2
import os
import h5py
import sys
import pickle
# fixed-sizes for image
fixed_size = tuple((256, 256))

# path to training data
path = "dataset/svm_reduced"

# bins for histogram
bins = 8

# train_test_split size
test_size = 0.10

# seed for reproducing same results
seed = 9

# BOVW for SIFT
def create_BOVW(path):
    try:
        kmeans = pickle.load(open('bovw_dict.sav', 'rb'))
    except:
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=50)
        descriptors =np.array(0)
        for file in os.listdir(path):
            print file
            try:
                image = cv2.imread(path+'/'+file)
            except:
                continue
            if image is None:
                continue
            image = cv2.resize(image,fixed_size)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray,None)
            if len(np.shape(descriptors)) == 0:
                descriptors = np.array(desc)
            else:
                descriptors = np.row_stack((descriptors, np.array(desc)))
            print np.shape(descriptors)
        batch_size = np.shape(descriptors)[0]/10
        kmeans = MiniBatchKMeans(n_clusters=300,batch_size=batch_size)
        kmeans.fit(np.array(descriptors))
        pickle.dump(kmeans, open('bovw_dict.sav','wb'))

    return kmeans

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


def fd_sift(image, bovw_dict):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray,None)
    hist = np.zeros([300,1])
    for d in desc:
        cluster = bovw_dict.predict(d.reshape(1,-1))
        hist[cluster] += 1
    return hist.flatten()

def fd_hog(image):
    hog = cv2.HOGDescriptor()
    img_squished = cv2.resize(image, (64,128))
    locations  = ((0,0),)
    h = hog.compute(image, locations = locations)
    return h.flatten()

def fd_orb(image):
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(img, kp)
    return kp

def fd_gist(image):
    return gist.extract(image)

def fd_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=16, R=2)
    return lbp.flatten()

active_fd = [1,1,1,1,1,0,1,0]
encoding = int("".join(map(str,active_fd)),base=2)
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
    bovw_dict = create_BOVW(path)
    checkpoint = 1
    image_count = len(os.listdir(path))
    # loop over the training data sub-folders
    # Expected number of checkpoints
    checkpoint_count = int(math.ceil(3000.0/ len(os.listdir(path))))
    print os.listdir('.')
    print 'checkpoint_' + str(encoding) + '_' + str(checkpoint_count) + '.npy'	
    if 'checkpoint_'+ str(encoding) + '_' + str(checkpoint_count) + '.npy' not in os.listdir('.'):
        for ind, file in enumerate(os.listdir(path)):
            if ind > 0 and ind % 3000 == 0:
                print np.shape(global_features)
                print sys.getsizeof(global_features)
                global_features = np.array(global_features)
                np.save('checkpoint_' +str(encoding) + '_' +  str(checkpoint), global_features.astype('float16'))
                # np.save('labels'+str(checkpoint),np.array(labels))
                if 'max_features' in dir():
                    max_features = np.max(np.vstack([max_features, global_features]), axis=0)
                    min_features = np.min(np.vstack([min_features, global_features]), axis=0)
                else:
                    max_features = np.max(global_features, axis=0)
                    min_features = np.min(global_features, axis=0)
                global_features = []
                # labels = []
                checkpoint = checkpoint + 1
            # get the current training label
            current_label = file.split('_')[0]
            ### DO LATER
            
            print file
            print "%.2f" % ((100*ind+0.0)/image_count)


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
            feature_list = []
            if active_fd[0]:
                feature_list.append(fd_hu_moments(image))
            if active_fd[1]:
                feature_list.append(fd_haralick(image))
            if active_fd[2]:
                feature_list.append(fd_histogram(image))
            if active_fd[3]:
                feature_list.append(fd_sift(image, bovw_dict))
            if active_fd[4]:
                feature_list.append(fd_hog(image))
            if active_fd[5]:
                feature_list.append(fd_orb(image))
            if active_fd[6]:
                feature_list.append(fd_gist(image))
            if active_fd[7]:
                feature_list.append(fd_lbp(image))
            

            ###################################
            # Concatenate global features
            ###################################
#            for feature in feature_list:
#		print feature.shape
	    global_feature = np.hstack(feature_list)

            # update the list of labels and feature vectors
            labels.append(current_label)
            global_features.append(global_feature)

        # Save the last few
        
        #print sys.getsizeof(global_features)
        np.save('checkpoint_' + str(encoding) + '_' + str(checkpoint), np.array(global_features))
        # np.save('labels'+str(checkpoint),np.array(labels))
        #print np.array(global_features).shape
        max_features = np.max( np.vstack([ max_features, np.array(global_features) ])  )	    
        global_features = []
        checkpoint = checkpoint + 1
    
    
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

    # # normalize the feature vector in the range (0-1)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit_transform(global_features)
    # #global_features = scaler.fit_transform(global_features)
    # print "[STATUS] feature vector normalized..."

    # print "[STATUS] target labels: {}".format(target)
    # print "[STATUS] target labels shape: {}".format(target.shape)
    
    for i in range(1, checkpoint):
        cp = np.load('checkpoint_%s_%s.npy' %(encoding, i) )
        cp = (cp - min_features) / (max_features - min_features)

        
        t = target[(i-1) * 3000: min(i * 3000, len(target))]
        # save the feature vector using HDF5
        h5f_data = h5py.File('output/data_%s_%s' % (str(encoding), i) + '.h5', 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(cp))
        h5f_data.close()

        h5f_label = h5py.File('output/labels_%s_%s' % (str(encoding), i) + '.h5', 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(t))
        h5f_label.close()


    print "[STATUS] end of training.."

    return global_features
