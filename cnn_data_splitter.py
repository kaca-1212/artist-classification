import os
import random as r

path = 'dataset/train'

# Split up the file names
labels = []
encodings = dict()
for file in os.listdir(path):
    artist = file.split('_')[0]
    if artist not in encodings:
        encodings[artist] = len(labels)
        labels.append([file])
    else:
        labels[encodings[artist]].append(file)

# Now choose random 10% to move to new path
# Splitting is done independently for each artist
split = 0.1
test_path = 'dataset/test'
for artist_code in list(encodings.values()):
    test_samples = r.sample(range(len(labels[artist_code])), int(len(labels[artist_code]) * split))
    for sample in test_samples:
        os.rename(path + '/' + labels[artist_code][sample], test_path + '/' + labels[artist_code][sample])