# simple cnn perform base class recognition
import sys
import numpy as np
from keras.models import Model
from keras.layers.core import *
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import io_data
import tensorflow as tf

np.random.seed()

data_path = sys.argv[1]
model_path = sys.argv[2]
sample = sys.argv[3]

n_class = 80
img_size = 32
sample = int(sample)


def shuffle(x, y):
	index = np.arange(x.shape[0])
	np.random.shuffle(index)
	x = x[index]
	y = y[index]

	return x, y


def validation(x, y):
	ratio = 0.1
	val_sample = int(ratio * x.shape[0])
	x_train = x[val_sample:]
	y_train = y[val_sample:]
	x_val = x[:val_sample]
	y_val = y[:val_sample]

	return x_train, y_train, x_val, y_val


def normalize(x):
	mean = 0.477513
	std = 0.267553

	return (x-mean)/std


def load_data(type = 'base'):
	novel_imgs, novel_label, base_imgs, base_label = io_data.read_train(data_path, sample = sample)

	base_label = [i for i in range(n_class) for j in range(500)]

	base_label = np_utils.to_categorical(base_label, n_class)

	base_imgs, base_label = shuffle(base_imgs, base_label)

	return base_imgs, base_label


def cnn():
	
	img_input = Input(shape = (img_size, img_size, 3))
	
	reg = l2(0.0005)
	
	# build Conv_net
	x = Conv2D(32,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block1_conv1')(img_input)
	x = Conv2D(32,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block1_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block1_pool')(x)
	x = Dropout(0.25)(x)

	x = Conv2D(64,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block2_conv1')(x)
	x = Conv2D(64,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block2_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block2_pool')(x)
	x = Dropout(0.25)(x)

	x = Conv2D(128,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block3_conv1')(x)
	x = Conv2D(128, (3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block3_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block3_pool')(x)
	x = Dropout(0.25)(x)

	#x = Conv2D(256,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block4_conv1')(x)
	#x = Conv2D(256, (3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block4_conv2')(x)
	#x = MaxPooling2D((2,2), name = 'block4_pool')(x)
	#x = Dropout(0.25)(x)
	
	x = Flatten()(x)

	x = Dense(units = 256, activation = 'elu', kernel_regularizer = reg, name = 'fc1')(x)
	x = Dropout(0.5)(x)
	out = Dense(units = n_class, activation = 'softmax', name = 'softmax')(x)
	model = Model(img_input, out)
	model.summary()
	
	return model


if __name__ == '__main__':

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)
	
	# first train on base class
	base_imgs, base_label = load_data()

	#base_imgs = normalize(base_imgs)
	#novel_imgs = normalize(novel_imgs)

	datagen = ImageDataGenerator(
	featurewise_center=False,  # set input mean to 0 over the dataset
	samplewise_center=False,  # set each sample mean to 0
	featurewise_std_normalization=False,  # divide inputs by std of the dataset
	samplewise_std_normalization=False,  # divide each input by its std
	zca_whitening=False,  # apply ZCA whitening
	rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=False) 
	
	# build model
	model1 = cnn()
	model1.compile(loss='categorical_crossentropy',optimizer = Adam(lr = 1.0e-3, decay = 1.0e-5),metrics=['accuracy'])
	checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True)
	earlystop = EarlyStopping(monitor = 'val_acc', patience = 20, verbose=0)
	callbacks = [earlystop, checkpoint]


	base_imgs, base_label, base_imgs_val, base_label_val = validation(base_imgs, base_label)
	
	# data_augmentation
	base_imgs, base_imgs_val, base_label, base_label_val = train_test_split(base_imgs, base_label, test_size = 0.1, random_state = 0)
	
	hist = model1.fit_generator(datagen.flow(base_imgs, base_label, batch_size = 128),
		steps_per_epoch = base_imgs.shape[0] / 128, epochs = 200, validation_data = (base_imgs_val, base_label_val), callbacks = callbacks)
	
	
	#hist = model1.fit(base_imgs, base_label, batch_size = 128, epochs = 300, shuffle = True, validation_split = 0.1, callbacks = callbacks)

	score = model1.evaluate(base_imgs, base_label)
	print('\nTrain Acc:', score[1])
	
	

