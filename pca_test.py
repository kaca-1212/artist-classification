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

train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)
param_grid = [
    {'C':[ .1,1,10,100,1000]}
]

print train_data.shape

model = LogisticRegression(C=10)#SVC(C=10, gamma=1, kernel='rbf')#GridSearchCV(LogisticRegression(), param_grid)
start_time = time.time()
model.fit(train_data, train_label)
train_time = time.time()-start_time
print "Train: " + str(train_time)
#print model.best_estimator_
#y_pred = model.predict(test_data)
#print confusion_matrix(test_label, y_pred)

start_time = time.time()
model.predict(np.array([test_data[0]]))
inf_time = time.time()-start_time
print "Inf: " + str(inf_time)
print "Accuracy:", model.score(train_data, train_label)
