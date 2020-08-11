import numpy as np
import argparse

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Embedding, Multiply, Activation, Conv2D, ZeroPadding2D, LocallyConnected2D, Concatenate, GRU, Lambda, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import UpSampling2D

import tensorflow as tf
import time
from tensorflow.keras import backend as K

print(tf.__version__)

loops_to_do = int(3E5)

# List of batch sizes to test
batch_size_i_array = np.logspace(0,4,10)

make_x_particles = False
loops = 5

GAN_noise_size = 128
GAN_output_size = 7
G_input = Input(shape=(GAN_noise_size,))

G = Dense(128, kernel_initializer='glorot_uniform')(G_input)
G = LeakyReLU(alpha=0.2)(G)
G = BatchNormalization()(G)
G = Reshape([8, 8, 2])(G)  # default: channel last
G = Conv2DTranspose(32, kernel_size=2, strides=1, padding="same")(G)
G = LeakyReLU(alpha=0.2)(G)
G = BatchNormalization()(G)
G = Conv2DTranspose(16, kernel_size=3, strides=1, padding="same")(G)
G = LeakyReLU(alpha=0.2)(G)
G = BatchNormalization()(G)
G = Flatten()(G)
G_output = Dense(GAN_output_size)(G)
G_output = Activation("tanh")(G_output)
generator = Model(G_input, G_output)
generator.summary()


for batch_size_i in batch_size_i_array:

	for i in range(0, 1):


		batch_size = int(batch_size_i)
		batchsize = int(batch_size_i)

		print(' ')
		print(batch_size)


		@tf.function(experimental_relax_shapes=True)
		def body(loop_index, output):

			noise = tf.random.normal((batchsize, 128), 0, 1)
			logits = generator([noise], training=False)

			return [loop_index+1, tf.concat([output, logits], axis=0)]


		print('warming up1')
		loop_index = tf.constant(0)

		output = tf.zeros([0, 7], dtype=tf.float32)
		
		condition_func = lambda loop_index, output: loop_index < 1
		t0 = time.time()
		generated_training = tf.while_loop(condition_func, body, loop_vars=[loop_index, output])[1]
		t1 = time.time()
		total_time = t1-t0
		print('warm up1',np.shape(generated_training),'time',total_time)



		print('warming up')
		loop_index = tf.constant(0)

		output = tf.zeros([0, 7], dtype=tf.float32)
		
		if make_x_particles == True:
			condition_func = lambda loop_index, output: tf.shape(output)[0] < loops_to_do
		elif make_x_particles == False:
			condition_func = lambda loop_index, output: loop_index < loops
		t0 = time.time()
		generated_training = tf.while_loop(condition_func, body, loop_vars=[loop_index, output])[1]
		t1 = time.time()
		total_time = t1-t0
		print('warm up',np.shape(generated_training),'time',total_time)



		print('Starting test...')
		loop_index = tf.constant(0)

		output = tf.zeros([0, 7], dtype=tf.float32)

		
		if make_x_particles == True:
			condition_func = lambda loop_index, output: tf.shape(output)[0] < loops_to_do
		elif make_x_particles == False:
			condition_func = lambda loop_index, output: loop_index < loops
		t0 = time.time()
		generated_training = tf.while_loop(condition_func, body, loop_vars=[loop_index, output])[1]
		t1 = time.time()
		print(np.shape(generated_training))

		total_time = t1-t0

		points_in_5_mins = (np.shape(generated_training)[0]/total_time)
		print('Time to generate %d points: %.3fs'%(np.shape(generated_training)[0],total_time))

		print('Points generated in 1 sec: %d'%int(points_in_5_mins),'batch_size',batch_size)


		''' Save results '''
		with open("results_dijet.txt", "a") as myfile:
			myfile.write('%d, %.2f \n'%(batch_size, points_in_5_mins))



