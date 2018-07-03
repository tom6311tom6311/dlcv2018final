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
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import io_data
from match import cosine_sim, euclidean_sim
from matching_model import *

np.random.seed()

# path
data_path = sys.argv[1]
model_path = sys.argv[2]
sample = sys.argv[3]

# parameters
img_size = 32
batch_size = 32
n_iterations = 1000000
display_steps = 10
patience = 10
sample = int(sample)
base_class = 80
novel_class = 20
base_examples = 500
novel_ratio = 1
finetune_iter = 0
nway = 20
n_supportset = nway*sample
average_per_class = True
fce = True


def data_augm(imgs):

	train_datagen = ImageDataGenerator(
		rotation_range = 15,
		width_shift_range = 0.1,
		height_shift_range = 0.1,
		shear_range = 0.1,
		zoom_range = 0.1,
		horizontal_flip = True,
		fill_mode = 'nearest')

	train_datagen.fit(imgs)

	return train_datagen



def load_data_batch(data, n_batch, n_iter, train_datagen, mode1 = 'train', mode2 = 'base'):
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
			classes = np.random.choice(base_class, nway, False)
			target_class = np.random.randint(0, nway)

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
			support_y.append(np_utils.to_categorical(slice_y, nway))
			target_y.append(np_utils.to_categorical(tar_y, nway))
		

	if mode1 == 'train' and mode2 == 'novel':
		for i in range(batch_size):
			slice_x = np.zeros((n_supportset , img_size ,img_size ,3))
			slice_x_tar = np.zeros((img_size ,img_size ,3))
			slice_y = np.zeros((n_supportset, ))

			ind = 0
			pinds = np.random.permutation(n_supportset)
			classes = np.random.choice(novel_class, nway, False)
			target_class = np.random.randint(0, nway)

			for j, cl in enumerate(classes):
				idx = np.arange(sample)
				for idx_ in idx:
					if np.random.randint(2, size=1) == 0:
						slice_x[pinds[ind],:,:,:] = train_datagen.random_transform(novel_imgs[cl][idx_])
					else:
						slice_x[pinds[ind],:,:,:] = novel_imgs[cl][idx_]
					slice_y[pinds[ind]] = j
					ind += 1

				if j == target_class:
					if np.random.randint(2, size=1) == 0:
						slice_x_tar = train_datagen.random_transform(novel_imgs[cl][np.random.choice(sample)])
					else:
						slice_x_tar = novel_imgs[cl][np.random.choice(sample)]
					tar_y = j

			support_x.append(slice_x)
			target_x.append(slice_x_tar)
			support_y.append(np_utils.to_categorical(slice_y, nway))
			target_y.append(np_utils.to_categorical(tar_y, nway))
		


	if mode1 == 'valid':
		for i in range(batch_size):
			slice_x = np.zeros((n_supportset , img_size ,img_size ,3))
			slice_x_tar = np.zeros((img_size ,img_size ,3))
			slice_y = np.zeros((n_supportset, ))

			ind = 0
			pinds = np.random.permutation(n_supportset)
			classes = np.random.choice(base_class, nway, False)
			target_class = np.random.randint(0, nway)

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
			support_y.append(np_utils.to_categorical(slice_y, nway))
			target_y.append(np_utils.to_categorical(tar_y, nway))
	
	support_x = np.array(support_x)
	support_y = np.array(support_y)
	target_x = np.array(target_x)
	target_y = np.array(target_y)

	return support_x, support_y, target_x, target_y



if __name__ == '__main__':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	novel_imgs, novel_label, base_imgs, base_label = io_data.read_train(data_path, sample = sample)

	# for novel data augmentation
	train_datagen = data_augm(novel_imgs)
	data = novel_imgs, novel_label, base_imgs, base_label

	matching = matching_net(sample_ = sample, average_per_class_ = average_per_class)

	matching.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 1.0e-6, decay = 1.0e-7), metrics=[categorical_accuracy])

	acc_lowest = 0.0
	
	for i in range(n_iterations):
		if (i+1) % novel_ratio == 0:
			support_x, support_y, target_x, target_y = load_data_batch(data, batch_size, i, train_datagen, 'train', 'novel')
		else:
			support_x, support_y, target_x, target_y = load_data_batch(data, batch_size, i, train_datagen, 'train')

		loss, acc = matching.train_on_batch([support_x, target_x, support_y], target_y)

		if i % display_steps == 0:
			support_x_val, support_y_val, target_x_val, target_y_val = load_data_batch(data, batch_size, i, train_datagen, 'valid')
			loss_val, acc_val = matching.test_on_batch([support_x_val, target_x_val, support_y_val], target_y_val)
			print("iteration {}, training loss: {:.3f}, training acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f}".format(i, loss, acc, loss_val, acc_val))

			# apply earlystopping
			if acc_val > acc_lowest:
				acc_lowest = acc_val
				patience_cnt = 0
				matching.save(model_path)
			else:
				patience_cnt += 1

			if patience_cnt > patience and i > 10:
				print('Early stopping...')
				break

	'''
	# fine tune on novel class
	matching.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = 1.0e-5), metrics=[categorical_accuracy])
	#matching.load_weight(model_path, by_name = True)
	for i in range(finetune_iter):
		support_x, support_y, target_x, target_y = load_data_batch(data, batch_size, train_datagen, 'train', 'novel')
		loss, acc = matching.train_on_batch([support_x, target_x, support_y], target_y)
		print("fine-tune iteration {}, training loss: {:.3f}, training acc: {:.3f}".format(i, loss, acc))

	matching.save(model_path)
	'''
