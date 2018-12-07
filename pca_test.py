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

features_test = ['gist'.'haralick','hist','hog','hu','SIFT']
enable = [1,1,1,1,1,1]

label_file = h5py.File('labels.h5','r')
labels = np.array(label_file['dataset_1'])

features = np.array([])

for i,feature in enumerate(features_test):
    if not enable[i]:
        continue
    feature_file = h5py.File(featire+'_features.h5','r')

    if features.size == 0:
        features = np.array(feature_file['dataset_1'])
    else:
        features = np.hstack([features, np.array(feature_file['dataset_1'])])
    
pca = PCA(n_components = 50)

features = pca.fit_transform(features)

param_grid = [
    {'C':[.00001,.0001,.001,0.01,1,10,100,1000,10000], 'gamma':[0.0001,0.001,0.01,0.1,1,10,100,1000], 'kernel':['rbf']}
]

model = GridSearchCV(SVC(), param_grid)

model.fit(train_data, train_label)
print model.best_estimator_
#y_pred = model.predict(test_data)
#print confusion_matrix(test_label, y_pred)
print "Accuracy:", model.score(test_data, test_label)