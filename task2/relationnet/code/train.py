import os, sys
import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train data path')
parser.add_argument('--test', help='test data path')
parser.add_argument('-l', '--log', help='log path')
parser.add_argument('-m', '--model', help='model path')
parser.add_argument('-s', '--sample', type=int, help='novel sample')
args = parser.parse_args()

log_path = args.log
model_path = args.model
train_path = args.train
test_path = args.test
sample = args.sample
n_batch = 200

def batch_generator(X, Y, batch=n_batch):
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, ])

    d = dict()
    uni = np.unique(Y)
    for u in uni:
        d[u] = np.array([i for i in range(Y.shape[0]) if Y[i] == u])

    while True:
        idx1 = np.random.choice(X.shape[0], int(batch/10))
        idx1 = np.reshape(np.vstack([idx1]*10), (-1))
        idx2 = np.random.choice(X.shape[0], batch)
        for b in range(int(batch/2)):
            idx2[b] = np.random.choice(d[Y[idx1[b]]], 1)    

        '''
        idx1 = np.random.choice(X.shape[0], int(batch/10))
        idx1 = np.reshape(np.vstack([idx1]*5), (-1))
        idx2 = np.random.choice(X.shape[0], int(batch/2))
        for b in range(int(batch/4)):
            idx2[b] = np.random.choice(d[Y[idx1[b]]], 1)    
        idx1 = np.reshape(np.vstack([idx1, idx2]), (-1))
        idx2 = np.reshape(np.vstack([idx2, idx1[:int(batch/2)]]), (-1))
        '''

        batch_X1 = X[idx1]
        batch_X2 = X[idx2]
        batch_Y = np.array(Y[idx1]==Y[idx2]).astype(int)

        yield [batch_X1, batch_X2], batch_Y

def episode_generator(X, Y, batch=n_batch):
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, ])

    d = dict()
    uni = np.unique(Y)
    for u in uni:
        d[u] = np.array([i for i in range(Y.shape[0]) if Y[i] == u])

    while True:
        C = np.random.choice(uni, 20)
        idx1=[]
        for c in range(20):
            idx1.append(np.random.choice(d[C[c]], sample))
            idx2 = [d[C[c]]]
        idx1 = np.reshape(np.vstack(idx1), (-1))
        idx1 = np.tile(idx1, (int(batch/(20*sample))))
        idx2 = np.reshape(np.vstack(idx2), (-1))
        idx2 = np.random.choice(idx2, batch)

        for b in range(int(batch/2)):
            idx2[b] = np.random.choice(d[Y[idx1[b]]], 1)    


        batch_X1 = X[idx1]
        batch_X2 = X[idx2]
        batch_Y = np.array(Y[idx1]==Y[idx2]).astype(int)

        yield [batch_X1, batch_X2], batch_Y


def evaluate_generator(X, Y, vX, vY, batch=n_batch):
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, height, width, channel])
    batch_X1 = np.zeros([batch, ])

    d = dict()
    uni = np.unique(Y)
    for u in uni:
        d[u] = np.array([i for i in range(Y.shape[0]) if Y[i] == u])

    while True:
        idx1 = np.random.choice(vX.shape[0], batch)
        idx2 = np.random.choice(X.shape[0], batch)
        for b in range(int(batch/2)):
            idx2[b] = np.random.choice(d[vY[idx1[b]]], 1)    

        '''
        idx1 = np.random.choice(vX.shape[0], int(batch/2))
        idx2 = np.random.choice(X.shape[0], int(batch/2))
        for b in range(int(batch/4)):
            idx2[b] = np.random.choice(d[vY[idx1[b]]], 1)    
        '''
        batch_X1 = vX[idx1]
        batch_X2 = X[idx2]
        #batch_X1 = np.reshape(np.vstack([batch_X1, batch_X2]), (-1, height, width, channel))
        #batch_X2 = np.reshape(np.vstack([batch_X2, batch_X1[:int(batch/2)]]), (-1, height, width, channel))

        batch_Y = np.array(vY[idx1]==Y[idx2]).astype(int)
        #batch_Y = np.reshape(np.vstack([batch_Y]*2), (-1))

        yield [batch_X1, batch_X2], batch_Y

def save_model(model, i=None):
    print('save model')
    if i:
        model.save_weights(os.path.join(model_path, 'RelationNet_sample_' + str(sample) + '_' + str(i) + '.h5'))
    else:
        model.save_weights(os.path.join(model_path, 'RelationNet_sample_' + str(sample) + '.h5'))

def save_log(loss, acc):
    with open(os.path.join(log_path, 'loss_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(loss, handle)
    with open(os.path.join(log_path, 'acc_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(acc, handle)
        
    
print('Read data')
novel_imgs, novel_label, base_imgs, base_label = read_train(train_path, sample=sample)

height, width, channel = novel_imgs.shape[1:]


# data augumentation
novel_imgs = np.vstack([novel_imgs, np.flip(novel_imgs, axis=1)])
novel_label = np.reshape(np.vstack([novel_label, novel_label]), (-1))

novel_imgs = np.vstack([novel_imgs, np.rot90(novel_imgs, 1, axes=(1, 2)), 
    np.rot90(novel_imgs, 2, axes=(1, 2)), np.rot90(novel_imgs, 3, axes=(1, 2))])
novel_label = np.reshape(np.vstack([novel_label]*4), (-1))

novel_imgs = np.tile(novel_imgs, (10, 1, 1, 1))
novel_label = np.tile(novel_label, (10))

# valid data split
base_imgs = np.reshape(base_imgs, (80, -1, height, width, channel))
base_label = np.reshape(base_label, (80, -1))

splite = 5
np.random.seed(2767057678)
print('train', np.random.get_state()[1][0])
valid_idx = np.random.choice(base_label.shape[1], int(base_label.shape[1]/splite), replace=False)
valid_imgs = np.reshape(base_imgs[:, valid_idx], (-1, height, width, channel))
valid_label = np.reshape(base_label[:, valid_idx], (-1, ))

#base_idx = [i for i in range(base_label.shape[1]) if i not in valid_idx]
base_idx = np.setdiff1d(np.arange(base_label.shape[1]), np.array(valid_idx)).tolist()
base_imgs = np.reshape(base_imgs[:, base_idx], (-1, height, width, channel))
base_label = np.reshape(base_label[:, base_idx], (-1, ))


# data augmentation
'''
base_imgs = np.vstack([base_imgs, np.flip(base_imgs, axis=1)])
base_label = np.reshape(np.vstack([base_label, base_label]), (-1))
base_imgs = np.vstack([base_imgs, np.rot90(base_imgs, 1, axes=(1, 2)), 
    np.rot90(base_imgs, 2, axes=(1, 2)), np.rot90(base_imgs, 3, axes=(1, 2))])
base_label = np.reshape(np.vstack([base_label]*4), (-1))
'''

imgs = np.vstack([novel_imgs, base_imgs])
label = np.concatenate([novel_label, base_label])

#imgs = base_imgs
#label = base_label

print(imgs.shape)
print(label.shape)
print(valid_imgs.shape)
print(valid_label.shape)

model, _, _ = RelationNet(height, width, channel)

his_loss = []
his_acc = []


def training(model=model):
    best_acc = 0
    
    
    for i in range(100+1):
        print('Iteration %i' % i)
        his = model.fit_generator(batch_generator(imgs, label), steps_per_epoch = 1000, epochs=1)
        #his = model.fit_generator(episode_generator(imgs, label), steps_per_epoch = 1000, epochs=1)
        print(his.history)
        his_loss.append(his.history['loss'])
        his_acc.append(his.history['acc'])

        his = model.evaluate_generator(evaluate_generator(imgs, label, valid_imgs, valid_label), steps=20)
        print('evaluate', his)
        save_model(model, i)
        if i % 5 == 0:
            if his[1] >= best_acc:
               best_acc = his[1] 
            else:
                break

        #if i>0 and i%10==0:
        #    backend.set_value(model.optimizer.lr, model.optimizer.lr/2)

    save_model(model)
    save_log(his_loss, his_acc)
    
    #model.load_weights(os.path.join(model_path, 'RelationNet_sample_5_52.h5'))

    for i in range(10):
        his = model.fit_generator(batch_generator(novel_imgs, novel_label), steps_per_epoch = 1, epochs=1)
        print(his.history)
        his_loss.append(his.history['loss'])
        his_acc.append(his.history['acc'])


        save_model(model, 'finetune_' + str(i))
    del model

training()
'''
try:
    training()
except:
    save_model(model)
    save_log(his_loss, his_acc)
    del model
'''
