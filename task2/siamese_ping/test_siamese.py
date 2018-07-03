# siamese test file
import os
import sys
import numpy as np
import pandas as pd
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
import tensorflow as tf
import io_data

# parameters
novel_class = 20
img_size = 32

def progress(count, total, suffix=''):
	bar_len = 60
	filled_len = int(round(bar_len * count / float(total)))
	percents = round(100.0 * count / float(total), 1)
	bar = '#' * filled_len + '-' * (bar_len - filled_len)
	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
	sys.stdout.flush()

def find_closest_class(sim, label):
	idx_sim_rank = np.argsort(sim)
	label_sim_rank = label[idx_sim_rank,]

	label_count = { 'null': 0 }
	curr_closest = 'null'
	for idx, l in enumerate(label_sim_rank):
		if l not in label_count:
			label_count[l] = idx
		else:
			label_count[l] += idx
		if label_count[l] >= label_count[curr_closest]:
			curr_closest = l
	return curr_closest


def compute_acc(pred, true):
	return np.mean((pred == true).astype(int))


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
	set_session(tf.Session(config=config))

	data_path = str(sys.argv[2])
	test_path = str(sys.argv[3])
	model_path = str(sys.argv[4])
	output_file = str(sys.argv[5])
	sample = int(sys.argv[6])

	siamese = load_model(model_path)

	novel_imgs, novel_label, test_imgs = io_data.read_test(data_path, test_path, sample = sample)

	pairs = [np.zeros((sample*novel_class, img_size, img_size, 3)) for i in range(2)]

	class_predictions = []
	print('Predicting...\n')
	for i in range(test_imgs.shape[0]):
		for j in range(sample*novel_class):
			pairs[0][j,:,:,:] = test_imgs[i].reshape(img_size, img_size, 3)
			pairs[1][j,:,:,:] = novel_imgs[j].reshape(img_size, img_size, 3)

		# compute similarity
		sim = siamese.predict(pairs).reshape((-1,))

		pred = find_closest_class(sim, novel_label)

		class_predictions.append(pred)
		progress(i+1, test_imgs.shape[0])
	print('\n')

	class_predictions = np.array(class_predictions)
	# acc = compute_acc(class_predictions, test_label)
	# print("Acc: ", acc)

	idx = np.arange(len(class_predictions))
	f = pd.DataFrame({'image_id':idx, 'predicted_label':class_predictions})
	f.to_csv(output_file, index = False)