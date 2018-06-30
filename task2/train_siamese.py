from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras import initializers
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
from sklearn.cross_validation import train_test_split
import sys
import os
import tensorflow as tf
import io_data
import numpy as np



def compute_accuracy(preds, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    print(preds)
    print(labels)
    return np.mean((np.around(preds) == labels).astype(int))


def create_base_network(input_d):
    w_init = initializers.TruncatedNormal(mean = 0.0, stddev = 1e-2)
    b_init = initializers.TruncatedNormal(mean = 0.5, stddev = 1e-2)

    seq = Sequential()
    nb_filter = [32, 64, 64, 128]
    kern_size = 3
    # conv layers
    seq.add(Conv2D(nb_filter[0], kern_size, kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-4), input_shape=input_d, padding='same'))
    seq.add(BatchNormalization())
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D((2, 2), strides = (2, 2)))
    seq.add(Dropout(.2))

    for nb_f in nb_filter[1:]:
        seq.add(Conv2D(nb_f, kern_size, kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-4), padding='same'))
        seq.add(BatchNormalization())
        seq.add(Activation('relu'))
        seq.add(MaxPooling2D((2, 2), strides = (2, 2)))
        seq.add(Dropout(.2))

    # dense layers
    seq.add(Flatten())
    seq.add(Dense(256, activation='sigmoid'))
    return seq

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    MODEL_PATH = str(sys.argv[2])

    img_pairs, labels = io_data.read_train_pairwise('data/', total_num=100000)

    print('training img pairs shape: ', img_pairs.shape)
    print('training labels shape: ', labels.shape)

    img_pairs_train, img_pairs_test, labels_train, labels_test = train_test_split(img_pairs, labels, test_size=.1)

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
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=30)])

    model.save(MODEL_PATH, include_optimizer=False)
    print('model saved')

    # compute final accuracy on training and test sets
    pred = model.predict([img_pairs_train[:, 0], img_pairs_train[:, 1]]).reshape((-1,))
    tr_acc = compute_accuracy(pred, labels_train)
    pred = model.predict([img_pairs_test[:, 0], img_pairs_test[:, 1]]).reshape((-1,))
    te_acc = compute_accuracy(pred, labels_test)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))