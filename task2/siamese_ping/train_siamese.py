from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Sequential, Model
from keras import initializers
from keras.optimizers import RMSprop
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import binary_accuracy
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
import sys
import os
import tensorflow as tf
import io_data
import numpy as np

NUM_BASE_CLASS = 80
NUM_NOVEL_CLASS = 20
NOVEL_SAMPLE = 5
NUM_BASE_EXAMPLES = 500
IMG_SIZE = 32
BATCH_SIZE = 128
MAX_EPOCH = 1000000
NOVEL_RATIO = 4
DISPLAY_RATIO = 301
PATIENCE = 40

def data_augm(imgs):
	train_datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest')
	train_datagen.fit(imgs)
	return train_datagen

def load_pair_batch(data, train_datagen, mode1 = 'train', mode2 = 'base'):
    novel_imgs, novel_label, base_imgs, base_label = data
    novel_imgs = novel_imgs.reshape((NUM_NOVEL_CLASS, NOVEL_SAMPLE, IMG_SIZE, IMG_SIZE, 3))
    base_imgs = base_imgs.reshape((NUM_BASE_CLASS, NUM_BASE_EXAMPLES, IMG_SIZE, IMG_SIZE, 3))

    if mode1 == 'train' and mode2 == 'base':
        # random choose category
        categories = np.random.choice(NUM_BASE_CLASS, size = (BATCH_SIZE,), replace = True)
        pairs = [np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)) for i in range(2)]
        targets = np.zeros((BATCH_SIZE,))
        same_idx = np.random.choice(BATCH_SIZE, size = (BATCH_SIZE//2,), replace = False)
        targets[same_idx] = 1
        for i in range(BATCH_SIZE):
            category = categories[i]
            idx_1 = np.random.randint(0, 400)
            pairs[0][i,:,:,:] = base_imgs[category, idx_1].reshape(IMG_SIZE, IMG_SIZE, 3)
            idx_2 = np.random.randint(0, 400)
            if i in same_idx:
                # same category
                category2 = category
            else:
                # different category
                category2 = (category + np.random.randint(1, NUM_BASE_CLASS)) % NUM_BASE_CLASS
                #print('train: ', category, category2)
            pairs[1][i,:,:,:] = base_imgs[category2, idx_2].reshape(IMG_SIZE, IMG_SIZE, 3)

    if mode1 == 'train' and mode2 == 'novel':
        # random choose category
        categories = np.random.choice(NUM_NOVEL_CLASS, size = (BATCH_SIZE,), replace = True)
        pairs = [np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)) for i in range(2)]
        targets = np.zeros((BATCH_SIZE,))
        same_idx = np.random.choice(BATCH_SIZE, size = (BATCH_SIZE//2,), replace = False)
        targets[same_idx] = 1
        for i in range(BATCH_SIZE):
            category = categories[i]
            idx_1 = np.random.randint(0, NOVEL_SAMPLE)
            tmp = novel_imgs[category, idx_1].reshape(IMG_SIZE, IMG_SIZE, 3)
            if np.random.randint(2, size=1) == 0:
                pairs[0][i,:,:,:] = train_datagen.random_transform(tmp)
            else:
                pairs[0][i,:,:,:] = tmp

            idx_2 = np.random.randint(0, NOVEL_SAMPLE)
            if i in same_idx:
                # same category
                category2 = category
            else:
                # different category
                category2 = (category + np.random.randint(1, NUM_NOVEL_CLASS)) % NUM_NOVEL_CLASS
                #print('train: ', category, category2)
            tmp = novel_imgs[category2, idx_2].reshape(IMG_SIZE, IMG_SIZE, 3)
            if np.random.randint(2, size=1) == 0:
                pairs[1][i,:,:,:] = train_datagen.random_transform(tmp)
            else:
                pairs[1][i,:,:,:] = tmp

    if mode1 == 'valid' and mode2 == 'base':
        # random choose category
        categories = np.random.choice(NUM_BASE_CLASS, size = (BATCH_SIZE,), replace = True)
        pairs = [np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)) for i in range(2)]
        targets = np.zeros((BATCH_SIZE,))
        same_idx = np.random.choice(BATCH_SIZE, size = (BATCH_SIZE//2,), replace = False)
        targets[same_idx] = 1
        for i in range(BATCH_SIZE):
            category = categories[i]
            idx_1 = np.random.randint(400, 500)
            pairs[0][i,:,:,:] = base_imgs[category, idx_1].reshape(IMG_SIZE, IMG_SIZE, 3)
            idx_2 = np.random.randint(400, 500)
            if i in same_idx:
                # same category
                category2 = category
            else:
                # different category
                category2 = (category + np.random.randint(1, NUM_BASE_CLASS)) % NUM_BASE_CLASS
                #print('valid: ', category, category2)
            pairs[1][i,:,:,:] = base_imgs[category2, idx_2].reshape(IMG_SIZE, IMG_SIZE, 3)

    if mode1 == 'valid' and mode2 == 'novel':
        # random choose category
        categories = np.random.choice(NUM_NOVEL_CLASS, size = (BATCH_SIZE,), replace = True)
        pairs = [np.zeros((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)) for i in range(2)]
        targets = np.zeros((BATCH_SIZE,))
        same_idx = np.random.choice(BATCH_SIZE, size = (BATCH_SIZE//2,), replace = False)
        targets[same_idx] = 1
        for i in range(BATCH_SIZE):
            category = categories[i]
            idx_1 = np.random.randint(0, NOVEL_SAMPLE)
            pairs[0][i,:,:,:] = novel_imgs[category, idx_1].reshape(IMG_SIZE, IMG_SIZE, 3)

            idx_2 = np.random.randint(0, NOVEL_SAMPLE)
            if i in same_idx:
                # same category
                category2 = category
            else:
                # different category
                category2 = (category + np.random.randint(1, NUM_NOVEL_CLASS)) % NUM_NOVEL_CLASS
                #print('train: ', category, category2)

            pairs[1][i,:,:,:] = novel_imgs[category2, idx_2].reshape(IMG_SIZE, IMG_SIZE, 3)

    return pairs, targets

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
    nb_filter = [32, 64, 128]
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

    DATA_PATH = str(sys.argv[2])
    MODEL_PATH = str(sys.argv[3])

    novel_imgs, novel_label, base_imgs, base_label = io_data.read_train(DATA_PATH, sample = NOVEL_SAMPLE)
    data = novel_imgs, novel_label, base_imgs, base_label

    train_datagen = data_augm(novel_imgs)

    input_dim = (IMG_SIZE, IMG_SIZE, 3)
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

    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=[binary_accuracy])

    curr_max_acc = 0.0

    for i in range(MAX_EPOCH):
        if i % NOVEL_RATIO == 0:
            imgs, target = load_pair_batch(data, train_datagen, 'train', 'novel')
        else:
            imgs, target = load_pair_batch(data, train_datagen, 'train', 'base')

        loss, acc = model.train_on_batch(imgs, target)

        if i % DISPLAY_RATIO == 0:
            imgs_val_base, target_val_base = load_pair_batch(data, train_datagen, 'valid', 'base')
            imgs_ori_novel, target_ori_novel = load_pair_batch(data, train_datagen, 'valid', 'novel')
            loss_val_base, acc_val_base = model.test_on_batch(imgs_val_base, target_val_base)
            loss_ori_novel, acc_ori_novel = model.test_on_batch(imgs_ori_novel, target_ori_novel)
            print("iteration {}, training loss: {:.3f}, training acc: {:.3f},base val loss: {:.3f}, base val acc: {:.3f},novel loss: {:.3f}, novel acc: {:.3f}".format(i, loss, acc, loss_val_base, acc_val_base, loss_ori_novel, acc_ori_novel))


            # apply earlystopping
            if acc_val_base > curr_max_acc:
                curr_max_acc = acc_val_base
                patience_cnt = 0
                model.save(MODEL_PATH)
            else:
                patience_cnt += 1

            if patience_cnt > PATIENCE and i > 20000:
                print('Early stopping...')
                break


    # # compute final accuracy on training and test sets
    # pred = model.predict([img_pairs_train[:, 0], img_pairs_train[:, 1]]).reshape((-1,))
    # tr_acc = compute_accuracy(pred, labels_train)
    # pred = model.predict([img_pairs_test[:, 0], img_pairs_test[:, 1]]).reshape((-1,))
    # te_acc = compute_accuracy(pred, labels_test)

    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    # print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
