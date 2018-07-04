# matching model
import os
import sys
import random
import numpy as np
from keras.layers import *
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras import initializers
from keras.models import load_model
import tensorflow as tf
from match import cosine_sim, euclidean_sim


def conv(input):
	reg = l2(0.0005)
	x = Conv2D(32,(3,3), activation = 'elu', padding = 'same', kernel_regularizer = reg, name = 'block1_conv1')(input)
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

	x = Flatten()(x)

	x = Dense(units = 256, activation = 'elu', kernel_regularizer = reg, name = 'fc1')(x)
	#x = Dense(units = 64, activation = 'elu', kernel_regularizer = reg, name = 'fc2')(x)

	return x


def matching_net(sample_ = 5, average_per_class_ = True, img_size = 32, batch_size = 32, nway = 20):
	n_supportset_ = sample_ * nway
	input1 = Input((n_supportset_, img_size, img_size ,3))
	input2 = Input((img_size, img_size ,3))

	tmp_in = Input((img_size, img_size, 3))
	conv_emb = conv(tmp_in)
	conv_net = Model(inputs = tmp_in, outputs = conv_emb)
	conv_net.load_weights('model/pretrained2.h5', by_name = True)
	#conv_net.load_weights('model/matching_pre2.h5', by_name = True)

	support_label = Input((n_supportset_, nway))

	inputs = []
	for lidx in range(n_supportset_):
			inputs.append(conv_net(Lambda(lambda x: x[:,lidx,:,:,:])(input1)))

	inputs.append(conv_net(input2))

	inputs.append(support_label)

	#out = euclidean_sim(nway = nway, sample = sample)(inputs)
	out = cosine_sim(nway = nway, sample = sample_, batch = batch_size, average_per_class = average_per_class_)(inputs)

	model = Model(inputs = [input1, input2, support_label], outputs = out)

	return conv_net, model