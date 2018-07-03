import os, sys
import numpy as np
import pickle
import argparse
from keras import backend
from keras.models import load_model
from keras.optimizers import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from model import *
from io_data import *
from sklearn.preprocessing import LabelEncoder

backend.set_image_dim_ordering('tf')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='train data path')
parser.add_argument('-l', '--log', help='log path')
parser.add_argument('-m', '--model', help='model path')
args = parser.parse_args()

log_path = args.log
model_path = args.model
train_path = args.train
n_batch = 400

def save_model(model):
    print('save model')
    model.save_weights(os.path.join(model_path, 'knn_model.h5'))

def save_log(loss, acc):
    with open(os.path.join(log_path, 'knn_loss_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(loss, handle)
    with open(os.path.join(log_path, 'knn_acc_'+str(sample)+'.log'), 'wb') as handle:
        pickle.dump(acc, handle)
        
np.random.seed(2926141147)
    
print('Read data')
imgs, label, test_imgs, test_label = read_pretrain(train_path)

_, height, width, channel = imgs.shape

le = LabelEncoder()
label = le.fit_transform(label)
test_label = le.transform(test_label)

# valid data split
valid_imgs = test_imgs
valid_label = test_label
imgs = np.vstack([imgs, np.flip(imgs, axis=-2)])
label = np.reshape(np.vstack([label]*2), (-1))

model, _ = Recognition(height, width, channel)

his_loss = []
his_acc = []


def training(model=model):
    best_loss = 10
    
    
    for i in range(100+1):
        print('Iteration %i' % i)
        his = model.fit(imgs, label, batch_size=n_batch, epochs=10)
        his_loss.append(his.history['loss'])
        his_acc.append(his.history['acc'])

        his = model.evaluate(valid_imgs, valid_label)

        if his[0] <= best_loss:
            best_loss = his[0] 
        else:
            break
        save_model(model)

    save_log(his_loss, his_acc)
    
    del model

training()
