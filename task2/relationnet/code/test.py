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
if evaluate == 1:
    train_imgs, label, test_imgs = read_test(train_path, test_path, sample=sample)
elif evaluate == 2:
    train_imgs, label, test_imgs, test_label = read_test2(train_path, test_path, sample=sample)

height, width, channel = train_imgs.shape[1:]

model, cnn_model, relation_model = RelationNet(height, width, channel)
model.load_weights(os.path.join(model_path, 'RelationNet_sample_' + str(sample) + '.h5'))

predict7 = []

train_imgs = cnn_model.predict(train_imgs, batch_size=100)
test_imgs = cnn_model.predict(test_imgs, batch_size=100)


T = train_imgs.shape[0]
for i in range(test_imgs.shape[0]):
    img = np.tile(np.expand_dims(test_imgs[i], axis=0), (T, 1, 1, 1))
    pro1 = np.squeeze(relation_model.predict([img, train_imgs], batch_size=100))
    pro2 = np.squeeze(relation_model.predict([train_imgs, img], batch_size=100))

    proI = pro1 + pro2

    # strategy 3
    idx = np.flip(np.argsort(proI), axis=-1)[:sample] #[:sample]
    predict_dict = dict()
    uni, count = np.unique(label[idx], return_counts=True)
    for u, c in zip(uni, count):
        predict_dict[u] = np.sum(proI[idx][label[idx]==u])
    predict7.append(max(predict_dict, key=lambda k: predict_dict[k]))
    
    save_predict(predict7, 'Relation_sample_'+str(sample)+'_predict.csv')

del model
    
