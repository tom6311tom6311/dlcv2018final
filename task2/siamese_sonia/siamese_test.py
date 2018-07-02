# siamese test file
import os
import sys
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import io_data

# path
data_path = sys.argv[1]
model_path = sys.argv[2]
output_path = sys.argv[3]

# parameters
sample = 5
novel_class = 20
img_size = 32


def find_max(sim, label):
	idx = np.argmax(sim)

	return label[idx]


def find_mean(sim, label):
	mean = []
	for i in range(novel_class):
		mean.append(np.mean(sim[i*sample:(i + 1)*sample]))

	idx = np.argmax(mean)*5

	return label[idx]


def compute_acc(pred, true):

	return np.mean((pred == true).astype(int))


if __name__ == '__main__':
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	siamese = load_model(model_path)

	#novel_imgs, novel_label, test_imgs = io_data.read_test(data_path, 'data/test', sample = sample)
	novel_imgs, novel_label, test_imgs, test_label = io_data.read_test2(data_path, os.path.join(data_path, 'novel'), sample = sample)

	pairs = [np.zeros((sample*novel_class, img_size, img_size, 3)) for i in range(2)]

	prediction = []
	for i in range(test_imgs.shape[0]):
		for j in range(sample*novel_class):
			pairs[0][j,:,:,:] = test_imgs[i].reshape(img_size, img_size, 3)
			pairs[1][j,:,:,:] = novel_imgs[j].reshape(img_size, img_size, 3)

		# compute similarity
		sim = siamese.predict(pairs)

		#pred = find_max(sim, novel_label)
		pred = find_mean(sim, novel_label)

		prediction.append(pred)

	prediction = np.array(prediction)
	acc = compute_acc(prediction, test_label)
	print("Acc: ", acc)
	'''
	f = pd.read_csv(output_path + 'sample_submission.csv')
	f['predicted_label'] = prediction
	f.to_csv(output_path + 'test.csv', index = False)
	'''