import os, sys
import numpy as np
import pandas as pd
import pickle
import argparse
from keras import backend
from keras.models import load_model
from keras.optimizers import *
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from model import *
from io_data import *

backend.set_image_dim_ordering('tf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train data path')
parser.add_argument('--test', help='test data path')
parser.add_argument('-l', '--log', help='log path')
parser.add_argument('-m', '--model', help='model path')
parser.add_argument('-s', '--sample', type=int, help='novel sample')
parser.add_argument('-e', '--evaluate', type=int, help='novel sample')
args = parser.parse_args()

log_path = args.log
model_path = args.model
train_path = args.train
test_path = args.test
sample = args.sample
evaluate = args.evaluate
width = 32
height= 32
channel = 3
n_batch = 100
epoch = 30
      
    
print('Read data')
train_imgs, label, test_imgs = read_test(train_path, test_path, sample=sample, random=True)
height, width, channel = train_imgs.shape[1:]

model, cnn_model = Recognition()
model.load_weights(model_path)

train_imgs = cnn_model.predict(train_imgs)
test_imgs = cnn_model.predict(test_imgs)

T = train_imgs.shape[0]

train_imgs = np.reshape(train_imgs, (20, sample, -1))
train_imgs = np.mean(train_imgs, axis=1)
label = np.array([label[i*sample] for i in range(20)])
test_imgs = np.reshape(test_imgs, (test_imgs.shape[0], -1))

knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(train_imgs, label)
predict = knc.predict(test_imgs)

save_predict(predict, 'knn_predict.csv')

uni, count = np.unique(predict,  return_counts=True)
for u, c in zip(uni, count):
    print(u, c)

del model
    
