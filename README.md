# Artist Identification with SVM and CNN

This repo contains the code used to perform artist identification of paintings in Python. Both an SVM and CNN approach are included. The SVM approach involves the extraction of multiple image feature descriptors, including the GIST descriptors, SIFT keypoints, and color histograms, while the CNN approach uses a simple CNN architecture to make predictions.

## Data
Data is not present in this repository for space concerns. The full dataset is made up of 7462 (256x256x3) images with corresponding labels, across 15 classes. 

## Feature Extraction
The feature extraction is run using the feature_extraction.py script. The script processes a set of images and labels, extracting a feature vector corresponding to each image descriptor for each sample. All feature vectors generated for each image descriptor are saved to separate .h5 files for loading and concatenation during the train script.

```
$ python feature_extraction.py
```

## SVM Training and Testing
The SVM training and validation is done using the pca_train_test.py script. The script splits the data into train and validation, performs the PCA on the chosen feature vectors for the training samples, and finds the best gamma and C parameters using 3-fold cross validation. The accuracy on the validation set is calculated to compare across different sets of features. The best models were tested on the test set by manually loading the test set and finding accuracy.

```
$ python pca_train_test.py
```

## CNN Training
The CNN is trained through the train_cnn.py script. This script defines the training parameters and model architecture, and runs data augmentation and back propagation to train the CNN. The weights are saved to be loaded in the future.

```
$ python train_cnn.py
```

## CNN Test
The CNN is tested through the run_cnn.py script. This script loads the saved weights from train_cnn.py and runs them on the test set to find accuracy and other metrics, as well as the confusion matrix.

```
$ python run_cnn.py
```

## Required Modules
Required modules include:
- h5py
- numpy
- opencv-contrib-python
- matplotlib
- scikit-learn
- scikit-image
- keras
- tensorflow
- seaborn
- pandas
- gist
- mahotas
- pickle
