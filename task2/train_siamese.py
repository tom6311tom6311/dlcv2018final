from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from sklearn.cross_validation import train_test_split
import sys
import os
import tensorflow as tf
import io_data
import numpy as np


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(preds, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    np.mean((preds == labels).astype(int))


def create_base_network(input_d):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    nb_filter = [32, 32]
    kern_size = 3
    # conv layers
    seq.add(Conv2D(nb_filter[0], kern_size, input_shape=input_d, padding='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
    seq.add(Dropout(.25))
    # conv layer 2
    seq.add(Conv2D(nb_filter[1], kern_size, padding='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))  # downsample
    seq.add(Dropout(.25))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(128, activation='sigmoid'))
    return seq

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    MODEL_PATH = str(sys.argv[2])

    img_pairs, labels = io_data.read_train_pairwise('data/')

    print('training img pairs shape: ', img_pairs.shape)
    print('training labels shape: ', labels.shape)

    img_pairs_train, img_pairs_test, labels_train, labels_test = train_test_split(img_pairs, labels, test_size=.1)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    input_dim = img_pairs_train.shape[2:]
    input_a = Input(shape=input_dim)
    input_b = Input(shape=input_dim)
    base_network = create_base_network(input_dim)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([processed_a, processed_b])

    out = Dense(1, activation = 'sigmoid')(L1_distance)

    model = Model(inputs=[input_a, input_b], outputs=out)

    print('\n\nbase network: \n')
    base_network.summary()
    print('\n\njoint network: \n')
    model.summary()

    # train
    epochs = 1000
    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms)
    xtr1 = img_pairs_train[:, 0]
    xtr2 = img_pairs_train[:, 1]
    model.fit(
        [xtr1, xtr2],
        labels_train,
        validation_split=.1,
        batch_size=128,
        epochs=epochs)
        # callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    model.save(MODEL_PATH, include_optimizer=False)
    print('model saved')

    # compute final accuracy on training and test sets
    pred = model.predict([img_pairs_train[:, 0], img_pairs_train[:, 1]])
    tr_acc = compute_accuracy(pred, labels_train)
    pred = model.predict([img_pairs_test[:, 0], img_pairs_test[:, 1]])
    te_acc = compute_accuracy(pred, labels_test)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))