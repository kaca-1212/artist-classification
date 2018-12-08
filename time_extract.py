import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import time

features_test = ['gist','haralick','hist','hog','hu','SIFT']


A = list(np.genfromtxt('svm_params.csv',delimiter=','))
A[0][0] = 1
A = list(A)
label_file = h5py.File('output/labels.h5','r')
labels = np.array(label_file['dataset_1'])
# 0 index is train time, 1 index is inference time
times = np.zeros([63,2])

for ind, combination in enumerate(A):
    enable = list(A[ind][0:6])
    C = A[ind][6]
    gamma = A[ind][7]

    features = np.array([])

    for i,feature in enumerate(features_test):
        if not enable[i]:
            continue
        feature_file = h5py.File('output/'+feature+'_features.h5','r')

        if features.size == 0:
            features = np.array(feature_file['dataset_1'])
        else:
            features = np.hstack([features, np.array(feature_file['dataset_1'])])
    labels = labels.flatten()
    print features.shape
    feature_count = min(features.shape[1], 100)
    pca = PCA(n_components = feature_count)

    #features = pca.fit_transform(features)

    (train_data, test_data, train_label, test_label) = train_test_split(features, labels, test_size=0.2)
    train_time_start = time.time()
    train_data = pca.fit_transform(train_data)
    test_data = pca.transform(test_data)
    #param_grid = [
    #    {'C':[ .1,1,10,100,1000], 'gamma':[.0001,.001,.01,.1,1,10], 'kernel':['rbf']}
    #]

    model = SVC(C=C, gamme=gamma, kernel='rbf')

    model.fit(train_data, train_label)
    train_time = time.time() - train_time_start
    inference_time_start = time.time()
    model.predict(test_data[0])
    inference_time = time.time() - inference_time_start

    times[ind][0] = train_time
    times[ind][1] = inference_time

np.savetxt('times.csv',times, delimiter=',')
