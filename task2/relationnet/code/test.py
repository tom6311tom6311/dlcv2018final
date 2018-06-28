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

# training imgs flip horizontally
'''
train_imgs = np.reshape(np.tile(train_imgs, (1, 2, 1, 1)), (-1, height, width, channel))
for i in range(sample):
    train_imgs[i*2] = np.flip(train_imgs[i*2], axis=0)
label =  np.reshape(np.tile(np.reshape(label, (-1, 1)), (1, 2)), (-1))
sample *= 2
train_imgs = np.reshape(np.tile(train_imgs, (1, 4, 1, 1)), (-1, height, width, channel))
for i in range(sample):
    for j in range(1, 4):
        train_imgs[i*4+j] = np.rot90(train_imgs[i*4+j], k=j, axes=(0, 1))
    #train_imgs[i*4+2] = np.rot90(train_imgs[i*4+2], k=2, axes=(0, 1))
    #train_imgs[i*4+3] = np.rot90(train_imgs[i*4+3], k=3, axes=(0, 1))
label =  np.reshape(np.tile(np.reshape(label, (-1, 1)), (1, 4)), (-1))

print(train_imgs.shape)
print(label.shape)
sample *= 4
'''

model, cnn_model, relation_model = RelationNet(height, width, channel)
#model.load_weights(os.path.join(model_path, 'RelationNet_sample_5_10.h5'))
model.load_weights(model_path)

train_imgs = cnn_model.predict(train_imgs)
test_imgs = cnn_model.predict(test_imgs)

predict1 = []
predict2 = []
predict5 = []
predict6 = []
label_mean = [label[i*sample] for i in range(int(label.shape[0]/sample))] 

T = train_imgs.shape[0]

for i in range(test_imgs.shape[0]):
    img = np.tile(np.expand_dims(test_imgs[i], axis=0), (T, 1, 1, 1))
    pro1 = relation_model.predict([img, train_imgs], batch_size=100)
    pro2 = relation_model.predict([train_imgs, img], batch_size=100)

    # strategy I
    proI = pro1+pro2

    #strategy II
    proII = pro1
    for p in range(proII.shape[0]):
        proII[p] = max([proII[p], pro2[p]])

    # strategy 1 
    # max
    predict1.append(label[np.argmax(proI)])
    predict5.append(label[np.argmax(proII)])

    # strategy 2
    # mean
    proI = np.reshape(proI, [-1, sample])
    proII = np.reshape(proII, [-1, sample])
    #print(pro.shape)
    proI = np.mean(proI, 1)
    proII = np.mean(proII, 1)
    predict2.append(label_mean[np.argmax(proI)])
    predict6.append(label_mean[np.argmax(proII)])

'''
train_imgs = np.reshape(train_imgs, (train_imgs.shape[0], -1))
test_imgs = np.reshape(test_imgs, (test_imgs.shape[0], -1))

pca = PCA(n_components=512)
pca.fit(np.vstack([train_imgs, test_imgs]))
train_pca = pca.transform(train_imgs)
test_pca = pca.transform(test_imgs)

knc = KNeighborsClassifier(n_neighbors=1)
knc.fit(train_pca, label)
predict3 = knc.predict(test_pca)

if sample != 1:
    knc = KNeighborsClassifier(n_neighbors=sample)
    knc.fit(train_pca, label)
    predict4 = knc.predict(test_pca)
else:
    predict4 = predict3
'''

if evaluate == 1:
    save_predict(predict1, 'I_max_predict.csv')
    save_predict(predict2, 'I_mean_predict.csv')
    #save_predict(predict3.tolist(), 'pca_knc1_predict.csv')
    #save_predict(predict4.tolist(), 'pca_knc_predict.csv')
    save_predict(predict5, 'II_max_predict.csv')
    save_predict(predict6, 'II_mean_predict.csv')


elif evaluate == 2:
    print('I max acc', accuracy_score(test_label, predict1))
    print('I mean acc', accuracy_score(test_label, predict2))
    #print('pca knc1 acc', accuracy_score(test_label, predict3))
    #print('pca knc acc', accuracy_score(test_label, predict4))
    print('II max acc', accuracy_score(test_label, predict5))
    print('II mean acc', accuracy_score(test_label, predict6))

del model
    
