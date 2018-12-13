#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

## REMEMBER TO AUGMENT DATA
# https://github.com/aleju/imgaug
# https://github.com/mdbloice/Augmentor


# import the necessary packages
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
from feature_extraction import get_feature_vectors
from sklearn.kernel_approximation import RBFSampler

# train_test_split size
test_size = 0.10
NUM_FILES = 3

# create all the machine learning models
# model = SVC(kernel='rbf', C = 200, gamma='auto', decision_function_shape='ovr')
model = SGDClassifier( loss='log',n_iter = 1000000)
# param_grid = [ {'C':[0.01,0.1,1,10,100,1000], 'gamma':['auto'],'kernel':['linear', 'rbf']} ]
#model = GridSearchCV(SVC(),param_grid = param_grid, cv=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42))
#model = LogisticRegression(C=10)
# model = LinearSVC(C=10, decision_function_shape='ovr')

# variables to hold the results and names
results = []
scoring = "accuracy"

# import the feature vector and trained labels

active_fd = [1,1,1,1,1,0,1,0]

feature_encoding = int("".join(map(str,active_fd)),base=2)

# Get all labels

all_labels = np.array([])
for i in range(1,NUM_FILES + 1):
    h5f_label = h5py.File('output/labels_%s_%s.h5' %(feature_encoding, i), 'r')
    all_labels = np.append(all_labels, np.array(h5f_label['dataset_1']))

all_labels = np.unique(all_labels)
all_test_labels = np.array([])

# Initialize RBF Approximator
rbf_feature = RBFSampler(gamma=1, n_components = 100, random_state = 1)

for i in range(1, NUM_FILES + 1):
    h5f_data = h5py.File('output/data_%s_%s.h5' %(feature_encoding, i), 'r')
    h5f_label = h5py.File('output/labels_%s_%s.h5' % (feature_encoding, i) , 'r')

    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']
	
    
    global_features =rbf_feature.fit_transform( np.array(global_features_string))
    global_labels = np.array(global_labels_string)
    
# use this if feature memory is too large to cast to np.array directly


# global_features = np.empty((0,len(h5f_data['dataset_1'][0])), dtype=np.float16)
# global_labels = np.empty((0,1), dtype=int)

# sample_proportion = 1
# print(np.unique(all_labels))
# for label in np.unique(all_labels)[0:5]:
#     class_coords = np.where(all_labels == label)[0]
#     samples = np.random.choice(class_coords, int(len(class_coords)*sample_proportion), replace =False)
#     global_features = np.row_stack((global_features, np.array(h5f_data['dataset_1'])[samples]))
#     global_labels = np.append(global_labels, all_labels[samples])
    
# h5f_data.close()
# h5f_label.close()

# verify the shape of the feature vector and labels
    print "[STATUS] features shape: {}".format(global_features.shape)
    print "[STATUS] labels shape: {}".format(global_labels.shape)



    print "[STATUS] training started..."

    # split all data randomly into 90% training and 10% testing data
    (trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                              np.array(global_labels),
                                                                                              test_size=test_size)

   
    
    if 'all_test_data' in dir():
	print all_test_data.shape
	all_test_data = np.vstack((all_test_data, testDataGlobal))
    else:
	all_test_data = testDataGlobal
    all_test_labels = np.append(all_test_labels, testLabelsGlobal)
    print "[STATUS] splitted train and test data..."
    print "Train data  : {}".format(trainDataGlobal.shape)
    print "Test data   : {}".format(testDataGlobal.shape)
    print "Train labels: {}".format(trainLabelsGlobal.shape)
    print "Test labels : {}".format(testLabelsGlobal.shape)

    # filter all the warnings
    import warnings
    warnings.filterwarnings('ignore')

    # # 10-fold cross validation
    # kfold = KFold(n_splits=10)
    # cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    # results.append(cv_results)
    # msg = "SVM: %f (%f)" % (cv_results.mean(), cv_results.std())
    # print msg





    model.partial_fit(trainDataGlobal, trainLabelsGlobal, classes = all_labels)
    # print model.cv_results_
    
y_pred = model.predict(all_test_data)
print confusion_matrix(all_test_labels, y_pred)
# path to test data
test_path = "dataset/test"

print "Accuracy:", model.score(testDataGlobal, testLabelsGlobal)

print metrics.classification_report(all_test_labels, y_pred)
