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
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import time


features_test = ['gist','haralick','hist','hog','hu','SIFT']
enable = [1,0,1,0,0,0]

label_file = h5py.File('output/labels.h5','r')
labels = np.array(label_file['dataset_1'])

features = np.array([])

# Load the features specified by enable
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
splitter =  StratifiedShuffleSplit(1, features, labels, test_size=0.1)
train_indices, test_indices = next(splitter.split(features, labels))

(train_data, test_data, train_label, test_label) =features[train_indices], labels[train_indices], features[test_indices], labels[test_indices]

# Fit the PCA on the training data, then additionally transform the test data
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)

# Form the parameter grid for the grid search
param_grid = [
    {'C':[ .00001, .0001, .001, .01,.1,1,10,100,1000, 10000], 'gamma':[.00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000], 'kernel':['rbf','linear']}
]

# The LR baseline
# model = LogisticRegression(C=10)
# The SVM after params are found from grid search
# model = SVC(C=10, gamma=1, kernel='rbf')
model = GridSearchCV(SVC(), param_grid)

# Time the training
start_time = time.time()
model.fit(train_data, train_label)
train_time = time.time()-start_time
print "Train time: " + str(train_time)

# Time the inference
start_time = time.time()
y_pred = model.predict(np.array([test_data[0]]))
inf_time = time.time()-start_time

# Show the confusion matrix
print confusion_matrix(test_label, y_pred)

print "Inf time: " + str(inf_time)
print "Accuracy:", model.score(train_data, train_label)
