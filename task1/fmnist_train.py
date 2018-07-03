import os
import sys
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
import skimage.io

num_classes = 10
img_size = 28 

data_path = sys.argv[1]
model_path = sys.argv[2]

def load_data():
	x_train = []
	y_train = []

	# load fashion mnist data
	path = os.path.join(data_path, 'train')
	folder = sorted(os.listdir(path))
	for i, f in enumerate(folder):
		path_ = os.path.join(path,f)
		images = sorted(os.listdir(path_))
		for image in images:
			img = skimage.io.imread(os.path.join(path_, image))
			x_train.append(img)
			y_train.append(i)
	x_train = np.array(x_train)
	y_train = np.array(y_train)

	# preprocess data, let pixel between 0~1
	x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
	x_train = x_train.astype('float32')/255

	y_train = np_utils.to_categorical(y_train, num_classes)

	# shuffle
	c = list(zip(x_train, y_train))
	np.random.shuffle(c)
	x_train, y_train = zip(*c)
	x_train = np.array(x_train)
	y_train = np.array(y_train)


	return x_train, y_train


def buildModel():
	img_input = Input(shape = (28, 28, 1))
	x = Conv2D(64,(3,3),activation = 'relu', padding = 'same', name = 'block1_conv1')(img_input)
	x = Conv2D(64,(3,3), activation = 'relu', padding = 'same', name = 'block1_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block1_pool')(x)
	x = Dropout(0.5, name = 'block1_drop')(x)

	model1 = Model(img_input, x)
	#model1.load_weights('model/mnist.h5', by_name = True)

	x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', name = 'block2_conv1')(model1.output)
	x = Conv2D(128,(3,3), activation = 'relu', padding = 'same', name = 'block2_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block2_pool')(x)
	x = Dropout(0.5, name = 'block2_drop')(x)

	x = Flatten()(x)
	x = Dense(units = 256, activation = 'relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(units = 256, activation = 'relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(units = 10,activation = 'softmax')(x)

	model = Model(img_input, x)
	model.summary()

	return model


if __name__ == '__main__':
	x_train, y_train = load_data()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	# build model
	model = buildModel()
	model.compile(loss='categorical_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
	checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True)
	earlystop = EarlyStopping(monitor='val_loss',patience=3,verbose=0)
	callbacks = [earlystop, checkpoint]
	hist = model.fit(x_train, y_train, batch_size = 128, epochs = 200, validation_split = 0.1, shuffle = True, callbacks = callbacks)

	score = model.evaluate(x_train,y_train)
	print('\nTrain Acc:', score[1])
