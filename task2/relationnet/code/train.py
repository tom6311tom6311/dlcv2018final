import os, sys
import numpy as np
import tensorflow as tf
import pickle
import argparse
from keras import backend
from keras.models import load_model
from keras.optimizers import *
from sklearn.metrics import accuracy_score
from model import *
from io_data import *

backend.set_image_dim_ordering('tf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train data path')
parser.add_argument('-l', '--log', help='log path')
parser.add_argument('-m', '--model', help='model path')
parser.add_argument('-s', '--sample', type=int, help='novel sample')
parser.add_argument('-p', dest='pretrain', default=False, action='store_true')
args = parser.parse_args()

log_path = args.log
model_path = args.model
train_path = args.train
sample = args.sample
pretrain = args.pretrain
n_batch = 200
repeat = 2 

def batch_generator(X, Y, batch=n_batch):
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, ])

    d = dict()
    uni = np.unique(Y)
    for u in uni:
        d[u] = np.array([i for i in range(Y.shape[0]) if Y[i] == u])

    while True:
        idx1 = np.random.choice(X.shape[0], int(batch/repeat))
        idx1 = np.reshape(np.vstack([idx1]*repeat), (-1))
        idx2 = np.random.choice(X.shape[0], batch)
        for b in range(int(batch/2)):
            idx2[b] = np.random.choice(d[Y[idx1[b]]], 1)    

        batch_X1 = X[idx1]
        batch_X2 = X[idx2]
        batch_Y = np.array(Y[idx1]==Y[idx2]).astype(int)

        yield [batch_X1, batch_X2], batch_Y

def save_model(model):
    print('save model')
    
    model.save_weights(os.path.join(model_path, 'RelationNet_sample_' + str(sample) + '.h5'))

def save_log(loss, acc, loss1, loss2):
    with open(os.path.join(log_path, 'loss_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(loss, handle)
    with open(os.path.join(log_path, 'acc_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(acc, handle)
    with open(os.path.join(log_path, 'loss1_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(loss1, handle)
    with open(os.path.join(log_path, 'loss2_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(loss2, handle)
        
    
print('Read data')
novel_imgs, novel_label, base_imgs, base_label = read_train(train_path, sample=sample)
novel_class = np.unique(novel_label)

height, width, channel = novel_imgs.shape[1:]

# valid data split
base_imgs = np.reshape(base_imgs, (80, -1, height, width, channel))
base_label = np.reshape(base_label, (80, -1))

splite = 5
print('train', np.random.get_state()[1][0])
valid_idx = np.random.choice(base_label.shape[1], int(base_label.shape[1]/splite), replace=False)
valid_imgs = np.reshape(base_imgs[:, valid_idx], (-1, height, width, channel))
valid_label = np.reshape(base_label[:, valid_idx], (-1, ))

#base_idx = [i for i in range(base_label.shape[1]) if i not in valid_idx]
base_idx = np.setdiff1d(np.arange(base_label.shape[1]), np.array(valid_idx)).tolist()
base_imgs = np.reshape(base_imgs[:, base_idx], (-1, height, width, channel))
base_label = np.reshape(base_label[:, base_idx], (-1, ))


imgs = np.vstack([novel_imgs, base_imgs])
label = np.concatenate([novel_label, base_label])

model, cnn, _ = RelationNet(height, width, channel)

his_loss = []
his_acc = []
his_loss1 = []
his_loss2 = []

def training(model=model):
    best_loss = 10
    b = 0

    for i in range(100+1):
        print('Iteration %i' % i)
        his = model.fit_generator(batch_generator(imgs, label), steps_per_epoch = 1000, epochs=1)
        his_loss.append(his.history['loss'])
        his_acc.append(his.history['acc'])
        his_loss1.append(his.history['loss1'])
        his_loss2.append(his.history['loss2'])

        his = model.evaluate_generator(batch_generator(valid_imgs, valid_label), steps=20)
        print('evaluate', his)
        if his[0] <= best_loss:
            best_loss = his[0]
            b = 0
            save_model(model)
        else:
            b += 1
            if b >= 5:
                break


    save_log(his_loss, his_acc, his_loss1, his_loss2)

    del model

training()
