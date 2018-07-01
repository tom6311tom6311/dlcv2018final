# siamese network
# modify from https://github.com/sorenbouma/keras-oneshot
import os
import sys
import random
import numpy as np
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import io_data
import matplotlib.pyplot as plt

# path
data_path = sys.argv[1]
model_path = sys.argv[2]

# parameters
img_size = 32
batch_size = 128
n_iterations = 1000000
display_steps = 301
patience = 30
sample = 5
base_class = 80
novel_class = 20
base_examples = 500
novel_ratio = 4


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

	return imgs, train_datagen



def load_pair_batch(data, n_batch, train_datagen, mode1 = 'train', mode2 = 'base'):
	novel_imgs, novel_label, base_imgs, base_label = data
	novel_imgs = novel_imgs.reshape((novel_class, sample, img_size, img_size, 3))
	base_imgs = base_imgs.reshape((base_class, base_examples, img_size, img_size, 3))
	
	if mode1 == 'train' and mode2 == 'base':
		# random choose category
		categories = np.random.choice(base_class, size = (batch_size,), replace = True)
		pairs = [np.zeros((batch_size, img_size, img_size, 3)) for i in range(2)]
		targets = np.zeros((batch_size,))
		same_idx = np.random.choice(batch_size, size = (batch_size//2,), replace = False)
		targets[same_idx] = 1
		for i in range(batch_size):
			category = categories[i]
			idx_1 = np.random.randint(0, 400)
			pairs[0][i,:,:,:] = base_imgs[category, idx_1].reshape(img_size, img_size, 3)
			idx_2 = np.random.randint(0, 400)
			if i in same_idx:
				# same category
				category2 = category
			else:
				# different category
				category2 = (category + np.random.randint(1, base_class)) % base_class
				#print('train: ', category, category2)
			pairs[1][i,:,:,:] = base_imgs[category2, idx_2].reshape(img_size, img_size, 3)


	if mode1 == 'train' and mode2 == 'novel':
		# random choose category
		categories = np.random.choice(novel_class, size = (batch_size,), replace = True)
		pairs = [np.zeros((batch_size, img_size, img_size, 3)) for i in range(2)]
		targets = np.zeros((batch_size,))
		same_idx = np.random.choice(batch_size, size = (batch_size//2,), replace = False)
		targets[same_idx] = 1
		for i in range(batch_size):
			category = categories[i]
			idx_1 = np.random.randint(0, sample)
			tmp = novel_imgs[category, idx_1].reshape(img_size, img_size, 3)
			if np.random.randint(2, size=1) == 0:
				pairs[0][i,:,:,:] = train_datagen.random_transform(tmp)
			else:
				pairs[0][i,:,:,:] = tmp

			idx_2 = np.random.randint(0, sample)
			if i in same_idx:
				# same category
				category2 = category
			else:
				# different category
				category2 = (category + np.random.randint(1, novel_class)) % novel_class
				#print('train: ', category, category2)
			tmp = novel_imgs[category2, idx_2].reshape(img_size, img_size, 3)
			if np.random.randint(2, size=1) == 0:
				pairs[1][i,:,:,:] = train_datagen.random_transform(tmp)
			else:
				pairs[1][i,:,:,:] = tmp


	if mode1 == 'valid':
		# random choose category
		categories = np.random.choice(base_class, size = (batch_size,), replace = True)
		pairs = [np.zeros((batch_size, img_size, img_size, 3)) for i in range(2)]
		targets = np.zeros((batch_size,))
		same_idx = np.random.choice(batch_size, size = (batch_size//2,), replace = False)
		targets[same_idx] = 1
		for i in range(batch_size):
			category = categories[i]
			idx_1 = np.random.randint(400, 500)
			pairs[0][i,:,:,:] = base_imgs[category, idx_1].reshape(img_size, img_size, 3)
			idx_2 = np.random.randint(400, 500)
			if i in same_idx:
				# same category
				category2 = category
			else:
				# different category
				category2 = (category + np.random.randint(1, base_class)) % base_class
				#print('valid: ', category, category2)
			pairs[1][i,:,:,:] = base_imgs[category2, idx_2].reshape(img_size, img_size, 3)


	return pairs, targets


def siamese_net():
	w_init = initializers.TruncatedNormal(mean = 0.0, stddev = 1e-2)
	b_init = initializers.TruncatedNormal(mean = 0.5, stddev = 1e-2)

	# build Conv_net
	img_input = Input(shape = (img_size, img_size, 3))
	x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same', 
		kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-4), name = 'conv1')(img_input)
	x = MaxPooling2D((2, 2), strides = (2, 2))(x)
	x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', 
		kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-4), name = 'conv2')(x)
	x = MaxPooling2D((2, 2), strides = (2, 2))(x)
	x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', 
		kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-4), name = 'conv3')(x)
	x = MaxPooling2D((2, 2), strides = (2, 2))(x)
	x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', 
		kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-4), name = 'conv4')(x)
	x = Flatten()(x)
	#x = Dense(512, activation = 'relu', kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-3))(x)
	out = Dense(256, activation = 'sigmoid', kernel_initializer = w_init, bias_initializer = b_init, kernel_regularizer=l2(2e-3), name = 'conv_out')(x)

	Conv_net = Model(inputs = img_input, outputs = out)
	#Conv_net.load_weights('model/pretrained.h5', by_name = True)

	# feed two input image to Conv_net
	input1 = Input(shape = (img_size, img_size, 3))
	input2 = Input(shape = (img_size, img_size, 3))

	out1 = Conv_net(input1)
	out2 = Conv_net(input2)

	# merge two output and predict similarity
	L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
	L1_distance = L1_layer([out1, out2])
	prediction = Dense(1, activation = 'sigmoid', bias_initializer = b_init)(L1_distance)
	siamese = Model(inputs = [input1, input2],outputs = prediction)

	Conv_net.summary()
	siamese.summary()
	
	return siamese


if __name__ == '__main__':

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	novel_imgs, novel_label, base_imgs, base_label = io_data.read_train(data_path, sample = sample)

	# for novel data augmentation
	novel_imgs, train_datagen = data_augm(novel_imgs)
	data = novel_imgs, novel_label, base_imgs, base_label

	siamese = siamese_net()
	siamese.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 1.0e-4), metrics=[binary_accuracy])

	loss_highest = 100.0

	for i in range(n_iterations):
		if i % novel_ratio == 0:
			image, target = load_pair_batch(data, batch_size, train_datagen, 'train', 'novel')
		else:
			image, target = load_pair_batch(data, batch_size, train_datagen, 'train')

		loss, acc = siamese.train_on_batch(image, target)

		if i % display_steps == 0:
			image_val, target_val = load_pair_batch(data, batch_size, train_datagen, 'valid')
			loss_val, acc_val = siamese.test_on_batch(image_val, target_val)
			print("iteration {}, training loss: {:.3f}, training acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f}".format(i, loss, acc, loss_val, acc_val))

			# apply earlystopping
			if loss_val < loss_highest:
				loss_highest = loss_val
				patience_cnt = 0
				siamese.save(model_path)
			else:
				patience_cnt += 1

			if patience_cnt > patience and i > 20000:
				print('Early stopping...')
				break
	
