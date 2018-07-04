# matching net test file
import os
import sys
import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf
import io_data
from match import cosine_sim, euclidean_sim
from matching_model import *

# path
data_path = sys.argv[1]
test_path = sys.argv[2]
model_path = sys.argv[3]
output_file = sys.argv[4]
sample = sys.argv[5]

# parameters
sample = int(sample)
novel_class = 20
img_size = 32
batch_size = 32
n_supportset = sample*novel_class
average_per_class = True


def compute_acc(pred, true):

	return np.mean((pred == true).astype(int))


if __name__ == '__main__':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	conv_net, matching = matching_net(sample_ = sample, average_per_class_ = average_per_class)
	matching.load_weights(model_path, by_name = True)

	novel_imgs, novel_label, test_imgs = io_data.read_test(data_path, test_path, sample = sample)

	support_x_ = novel_imgs

	tmp = [i for i in range(novel_class) for j in range(sample)]
	support_y_ = np_utils.to_categorical(tmp, novel_class)

	prediction = []
	for i in range(int(test_imgs.shape[0]/batch_size) + 1):
		target_x = test_imgs[batch_size*i: min(test_imgs.shape[0], batch_size*(i+1))]
		batch = target_x.shape[0]
		support_x = np.array([support_x_ for i in range(batch)])
		support_y = np.array([support_y_ for i in range(batch)])

		if batch != batch_size:
			support_x = np.vstack((support_x, support_x))
			target_x = np.vstack((target_x, target_x))
			support_y = np.vstack((support_y, support_y))
			batch_ = batch_size
			pred = matching.predict([support_x, target_x, support_y], batch_size = batch_)
			pred = pred[0:batch]
		else:
			pred = matching.predict([support_x, target_x, support_y], batch_size = batch)

		prediction.append(pred)

	novel_label = novel_label[np.arange(0, novel_label.shape[0], sample)]

	prediction = np.vstack(prediction)
	idx = np.argmax(prediction, axis = 1)
	prediction = [novel_label[i] for i in idx]
	
	idx = np.arange(len(prediction))
	f = pd.DataFrame({'image_id':idx, 'predicted_label':prediction})
	f.to_csv(output_file, index = False)
	
