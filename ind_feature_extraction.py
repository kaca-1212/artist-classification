from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
import gist
import numpy as np
import mahotas
import cv2
import os
import h5py
import sys
import pickle
import time

data_path = 'dataset/svm_reduced'

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
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
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


def fd_gist(image):
    return gist.extract(image)

label_dict = {
    'Albrecht Durer': 0,
    'Boris Kustodiev': 1,
    'Camille Corot': 2,
    'Camille Pissarro': 3,
    'Childe Hassam': 4,
    'Claude Monet': 5,
    'Edgar Degas': 6,
    'Eugene Boudin': 7,
    'Giovanni Battista Piranesi': 8,
    'Gustave Dore': 9,
    'Henri Matisse': 10,
    'Ilya Repin': 11,
    'Ivan Aivazovsky': 12,
    'Ivan Shishkin': 13,
    'John Singer Sargent': 14
}

kmeans = create_BOVW(data_path)
num_images = len(os.listdir(data_path))

avg_extraction_times = np.zeros([1,6])

labels = np.array([])
features = np.array([])
# Do sift separately
start_time = time.time()
for ind, f in enumerate(os.listdir(data_path)):
    print (ind + 0.0)/num_images
    label = label_dict[f.split('_')[0]]
    if labels.size == 0:
        labels = np.array([label])
    else:
        labels = np.vstack((labels, label))

    img = cv2.imread(data_path + '/' + f)

    if features.size == 0:
        features = fd_sift(img,kmeans)
    else:
        features = np.vstack((features, fd_sift(img,kmeans)))
extraction_time = time.time() - start_time
avg_time = extraction_time/num_images
avg_extraction_times[0] = avg_time
"""
f_save = h5py.File('output/SIFT_features.h5','w')
f_save.create_dataset('dataset_1',data=features)
l_save = h5py.File('output/SIFT_labels.h5','w')
l_save.create_dataset('dataset_1',data=labels)
f_save.close()
l_save.close()
"""

feature_functions = [fd_histogram, fd_hog, fd_gist, fd_hu_moments, fd_haralick]
feature_names = ['hist','hog','gist''hu', 'haralick',]
# For each feature type


for ind, func in enumerate(feature_functions):
    labels = np.array([])
    features = np.array([])
    
    start_time = time.time()

    for ind2, f in enumerate(os.listdir(data_path)):
        print (ind2 + 0.0)/num_images
        label = label_dict[f.split('_')[0]]
        if labels.size == 0:
            labels = np.array([label])
        else:
            labels = np.vstack((labels, label))

        img = cv2.imread(data_path + '/' + f)

        if features.size == 0:
            features = func(img)
        else:
            features = np.vstack((features, func(img)))
    extraction_time = time.time() - start_time
    avg_time = extraction_time/num_images
    avg_extraction_times[ind + 1] = avg_time
    """
    f_save = h5py.File('output/' + feature_names[ind] + '_features.h5','w')
    f_save.create_dataset('dataset_1',data = features)
    f_save.close()
    l_save = h5py.File('output/' + feature_names[ind] + '_labels.h5','w')
    l_save.create_dataset('dataset_1',data = labels)
    l_save.close()
    """

np.savetxt('extraction_times.csv', avg_extraction_times, delimiter=',')
