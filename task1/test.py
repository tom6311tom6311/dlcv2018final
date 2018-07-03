import os
import sys
import numpy as np
import pandas as pd
import skimage.io
import keras.backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf
from keras.utils import np_utils
from keras.datasets import fashion_mnist

data_path = sys.argv[1]
model_path = sys.argv[2]
output_file = sys.argv[3]

img_size = 28
test_num = 10000

def load_data():
	x_test = []
	# load fashion mnist data
	path = os.path.join(data_path, 'test')
	for image in range(test_num):
		img = skimage.io.imread(os.path.join(path, str(image) + '.png'))
		x_test.append(img)
	x_test = np.array(x_test)

	# preprocess data, let pixel between 0~1
	x_test = x_test.reshape(x_test.shape[0], img_size, img_size, 1)
	x_test = x_test.astype('float32')/255

	return x_test


if __name__ == '__main__':
	x_test = load_data()

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)
	K.set_session(session)

	# predict
	model = load_model(model_path)
	model.summary()
	y_pred = model.predict(x_test, batch_size = 128)
	y_pred = np.argmax(y_pred, axis = -1)

	idx = np.arange(len(y_pred))
	f = pd.DataFrame({'image_id':idx, 'predicted_label':y_pred})
	f.to_csv(output_file, index = False)


