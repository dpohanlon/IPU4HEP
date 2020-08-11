import numpy as np
import argparse

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Embedding, Multiply, Activation, Conv2D, ZeroPadding2D, LocallyConnected2D, Concatenate, GRU, Lambda
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
batch_size_i_array = np.logspace(0,np.log10(25000),10)


make_x_particles = False
loops = 100

''' Create the network. '''
# NON-PROMPT
generator = Sequential()

generator.add(Dense(1536, input_shape=(100,)))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(3072))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(6, activation='tanh'))
generator.add(Reshape((1, 6, 1)))


# PROMPT
# generator = Sequential()
# generator.add(Dense(512, input_shape=(100,)))
# generator.add(LeakyReLU(alpha=0.2))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Dense(1024))
# generator.add(LeakyReLU(alpha=0.2))
# generator.add(BatchNormalization(momentum=0.8))
# generator.add(Dense(4, activation='tanh'))
# generator.add(Reshape((1, 4, 1)))

generator.summary()



for batch_size_i in batch_size_i_array:

	batch_size = int(batch_size_i)
	batchsize = int(batch_size_i)

	print(' ')
	print(batch_size)



	@tf.function(experimental_relax_shapes=True)
	def body(loop_index, output):

		noise = tf.random.normal((batchsize, 100), 0, 1)
		logits = generator(noise, training=False)

		combined_output = tf.squeeze(logits,axis=-1)
		combined_output = tf.squeeze(combined_output,axis=1)

		return [loop_index+1, tf.concat([output, combined_output], axis=0)]


	print(loops_to_do/batchsize)


	print('Warming up...')
	loop_index = tf.constant(0)

	output = tf.zeros([0, 6], dtype=tf.float32)
	
	condition_func = lambda loop_index, output: loop_index < 1
	t0 = time.time()
	generated_training = tf.while_loop(condition_func, body, loop_vars=[loop_index, output])[1]
	t1 = time.time()
	total_time = t1-t0
	print('warm up1',np.shape(generated_training),'time',total_time)



	print('Starting test...')
	loop_index = tf.constant(0)

	output = tf.zeros([0, 6], dtype=tf.float32)

	
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

	print('Points generated in 5 mins: %d'%int(points_in_5_mins),'batch_size',batch_size)


	with open("results_shiplarge.txt", "a") as myfile:
		myfile.write('%d, %.2f \n'%(batch_size, points_in_5_mins))



