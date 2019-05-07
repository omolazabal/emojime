
import sys
sys.path.append("..")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from emojime.utils import plot_confusion_matrix
import pickle


data_path = '../data'
emotions = ['neutral', 'happy', 'sad', 'fear', 'angry']
data = np.load('../data/data_set.npy')

X = data[:,:-1]
y = data[:,-1]

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X, y = shuffle(X, y)

svm = SVC(probability=True)
svm.fit(X, y)

pickle.dump(svm, open('../models/svm_emotion_classifier', 'wb'))
pickle.dump(scaler, open('../models/emotion_scaler', 'wb'))
