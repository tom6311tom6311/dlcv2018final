# match fuinction for matching network
# modify from https://github.com/cnichkawde/MatchingNetwork
import tensorflow as tf
from keras.layers.merge import _Merge


class cosine_sim(_Merge):
	def __init__(self, novel_class = 20, sample = 5, **kwargs):
		super(cosine_sim,self).__init__(**kwargs)
		self.eps = 1e-10
		self.novel_class = novel_class
		self.sample = sample
		self.n_supportset = self.novel_class*self.sample

	def build(self, input_shape):
		if not isinstance(input_shape, list):
			raise ValueError('A CosineSim layer should be called on a list of inputs of support_emb, target_emb, support_label')

	def call(self, inputs):
		support_emb = inputs[:-2]
		target_emb = inputs[-2]
		support_label = inputs[-1]

		similarities = []
		sum_target = tf.reduce_sum(tf.square(target_emb), 1, keep_dims = True)
		targetmagnitude = tf.rsqrt(tf.clip_by_value(sum_target, self.eps, float("inf")))

		for i in range(self.n_supportset):
			emb = support_emb[i]
			sum_support = tf.reduce_sum(tf.square(emb), 1, keep_dims=True)
			supportmagnitude = tf.rsqrt(tf.clip_by_value(sum_support, self.eps, float("inf")))

			dot_product = tf.matmul(tf.expand_dims(target_emb,1),tf.expand_dims(emb,2))
			dot_product = tf.squeeze(dot_product,[1])

			sim = dot_product*supportmagnitude*targetmagnitude
			similarities.append(sim)

		similarities = tf.concat(axis = 1,values = similarities)
		softmax_similarities = tf.nn.softmax(similarities)
		preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities,1),support_label))

		preds.set_shape((support_emb[0].shape[0], self.novel_class))

		return preds

	def compute_output_shape(self,input_shape):
		input_shapes = input_shape
		return (input_shapes[0][0], self.novel_class)