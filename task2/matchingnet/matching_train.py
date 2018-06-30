# matching network
# modify from https://github.com/cnichkawde/MatchingNetwork
import os
import sys
import random
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import SGD,Adam
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import io_data
from match import cosine_sim

# path
data_path = sys.argv[1]
model_path = sys.argv[2]

# parameters
img_size = 32
batch_size = 32
n_iterations = 1000000
display_steps = 10
patience = 10
sample = 5
base_class = 80
novel_class = 20
base_examples = 500
novel_ratio = 1.0e10
finetune_iter = 20
n_supportset = novel_class*sample


def data_augm(imgs):

	train_datagen = ImageDataGenerator(
		rotation_range = 20,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True,
		fill_mode = 'nearest')

	train_datagen.fit(imgs)

	return train_datagen



def load_data_batch(data, n_batch, train_datagen, mode1 = 'train', mode2 = 'base'):
	novel_imgs, novel_label, base_imgs, base_label = data
	novel_imgs = novel_imgs.reshape((novel_class, sample, img_size, img_size, 3))
	base_imgs = base_imgs.reshape((base_class, base_examples, img_size, img_size, 3))
	
	support_x = []
	support_y = []
	target_x = []
	target_y = []

	if mode1 == 'train' and mode2 == 'base':
		for i in range(batch_size):
			slice_x = np.zeros((n_supportset , img_size ,img_size ,3))
			slice_x_tar = np.zeros((img_size ,img_size ,3))
			slice_y = np.zeros((n_supportset, ))

			ind = 0
			pinds = np.random.permutation(n_supportset)
			classes = np.random.choice(base_class, novel_class, False)
			target_class = np.random.randint(0, novel_class)

			for j, cl in enumerate(classes):
				idx = np.random.choice(400, size = (sample,), replace = False)
				for idx_ in idx:
					slice_x[pinds[ind],:,:,:] = base_imgs[cl][idx_]
					slice_y[pinds[ind]] = j
					ind += 1

				if j == target_class:
					slice_x_tar = base_imgs[cl][np.random.choice(400)]
					tar_y = j

			support_x.append(slice_x)
			target_x.append(slice_x_tar)
			support_y.append(np_utils.to_categorical(slice_y, novel_class))
			target_y.append(np_utils.to_categorical(tar_y, novel_class))
		

	if mode1 == 'train' and mode2 == 'novel':
		for i in range(batch_size):
			slice_x = np.zeros((n_supportset , img_size ,img_size ,3))
			slice_x_tar = np.zeros((img_size ,img_size ,3))
			slice_y = np.zeros((n_supportset, ))

			ind = 0
			pinds = np.random.permutation(n_supportset)
			classes = np.arange(novel_class)
			target_class = np.random.randint(0, novel_class)

			for j, cl in enumerate(classes):
				idx = np.arange(sample)
				for idx_ in idx:
					if np.random.randint(2, size=1) == 0:
						slice_x[pinds[ind],:,:,:] = train_datagen.random_transform(base_imgs[cl][idx_])
					else:
						slice_x[pinds[ind],:,:,:] = base_imgs[cl][idx_]
					slice_y[pinds[ind]] = j
					ind += 1

				if j == target_class:
					if np.random.randint(2, size=1) == 0:
						slice_x_tar = train_datagen.random_transform(base_imgs[cl][np.random.choice(sample)])
					else:
						slice_x_tar = base_imgs[cl][np.random.choice(sample)]
					tar_y = j

			support_x.append(slice_x)
			target_x.append(slice_x_tar)
			support_y.append(np_utils.to_categorical(slice_y, novel_class))
			target_y.append(np_utils.to_categorical(tar_y, novel_class))
		


	if mode1 == 'valid':
		for i in range(batch_size):
			slice_x = np.zeros((n_supportset , img_size ,img_size ,3))
			slice_x_tar = np.zeros((img_size ,img_size ,3))
			slice_y = np.zeros((n_supportset, ))

			ind = 0
			pinds = np.random.permutation(n_supportset)
			classes = np.random.choice(base_class, novel_class, False)
			target_class = np.random.randint(0, novel_class)

			for j, cl in enumerate(classes):
				idx = np.random.choice(100, size = (sample,), replace = False)
				for idx_ in idx:
					slice_x[pinds[ind],:,:,:] = base_imgs[cl][idx_ + 400]
					slice_y[pinds[ind]] = j
					ind += 1

				if j == target_class:
					slice_x_tar = base_imgs[cl][400 + np.random.choice(100)]
					tar_y = j

			support_x.append(slice_x)
			target_x.append(slice_x_tar)
			support_y.append(np_utils.to_categorical(slice_y, novel_class))
			target_y.append(np_utils.to_categorical(tar_y, novel_class))
		

	return np.array(support_x), np.array(support_y), np.array(target_x), np.array(target_y)


def conv(input):
	x = Conv2D(64,(3,3), activation = 'elu', padding = 'same', name = 'block1_conv1')(input)
	x = Conv2D(64,(3,3), activation = 'elu', padding = 'same', name = 'block1_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block1_pool')(x)
	x = Dropout(0.5)(x)

	x = Conv2D(128,(3,3), activation = 'elu', padding = 'same', name = 'block2_conv1')(x)
	x = Conv2D(128,(3,3), activation = 'elu', padding = 'same', name = 'block2_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block2_pool')(x)
	x = Dropout(0.5)(x)

	x = Conv2D(256,(3,3), activation = 'elu', padding = 'same', name = 'block3_conv1')(x)
	x = Conv2D(256, (3,3), activation = 'elu', padding = 'same', name = 'block3_conv2')(x)
	x = MaxPooling2D((2,2), name = 'block3_pool')(x)
	x = Dropout(0.5)(x)

	x = Flatten()(x)

	return x


def matching_net():
	input1 = Input((n_supportset, img_size, img_size ,3))
	input2 = Input((img_size, img_size ,3))

	tmp_in = Input((img_size, img_size, 3))
	conv_emb = conv(tmp_in)
	conv_net = Model(inputs = tmp_in, outputs = conv_emb)

	inputs = []
	for lidx in range(n_supportset):
		#inputs.append(conv_net(input1[:,lidx,:,:,:]))
		inputs.append(conv_net(Lambda(lambda x: x[:,lidx,:,:,:])(input1)))
	#support_emb = K.stack(support_emb, axis = 0)

	inputs.append(conv_net(input2))

	support_label = Input((n_supportset, novel_class))
	inputs.append(support_label)

	#out = Lambda(cosine_sim)([support_emb, target_emb, support_label])

	out = cosine_sim(novel_class = novel_class, sample = sample)(inputs)

	model = Model(inputs = [input1, input2, support_label], outputs = out)

	model.summary()

	return model


if __name__ == '__main__':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	novel_imgs, novel_label, base_imgs, base_label = io_data.read_train(data_path, sample = sample)

	# for novel data augmentation
	train_datagen = data_augm(novel_imgs)
	data = novel_imgs, novel_label, base_imgs, base_label

	matching = matching_net()
	matching.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 1.0e-4), metrics=[categorical_accuracy])

	acc_lowest = 0.0

	for i in range(n_iterations):
		if (i+1) % novel_ratio == 0:
			support_x, support_y, target_x, target_y = load_data_batch(data, batch_size, train_datagen, 'train', 'novel')
		else:
			support_x, support_y, target_x, target_y = load_data_batch(data, batch_size, train_datagen, 'train')

		loss, acc = matching.train_on_batch([support_x, target_x, support_y], target_y)

		if i % display_steps == 0:
			support_x_val, support_y_val, target_x_val, target_y_val = load_data_batch(data, batch_size, train_datagen, 'valid')
			loss_val, acc_val = matching.test_on_batch([support_x_val, target_x_val, support_y_val], target_y_val)
			print("iteration {}, training loss: {:.3f}, training acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f}".format(i, loss, acc, loss_val, acc_val))

			# apply earlystopping
			if acc_val > acc_lowest:
				acc_lowest = acc_val
				patience_cnt = 0
				matching.save(model_path)
			else:
				patience_cnt += 1

			if patience_cnt > patience and i > 20000:
				print('Early stopping...')
				break

	# fine tune on novel class
	#matching.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 1.0e-5), metrics=[categorical_accuracy])
	#matching.load_weight(model_path, by_name = True)
	for i in range(finetune_iter):
		support_x, support_y, target_x, target_y = load_data_batch(data, batch_size, train_datagen, 'train', 'novel')
		loss, acc = matching.train_on_batch([support_x, target_x, support_y], target_y)
		print("fine-tune iteration {}, training loss: {:.3f}, training acc: {:.3f}".format(i, loss, acc))

	matching.save(model_path)
