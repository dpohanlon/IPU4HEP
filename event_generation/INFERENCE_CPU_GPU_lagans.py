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
batch_size_i_array = np.logspace(0,4,10)


make_x_particles = False
loops = 100


''' Create the network. '''
latent_size = 200

loc = Sequential([
    Dense(128 * 7 * 7, input_dim=latent_size),
    Reshape((7, 7, 128)),

    Conv2D(64, (5, 5), padding='same', kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    UpSampling2D(size=(2, 2),interpolation='bilinear'),

    ZeroPadding2D((2, 2)),
    LocallyConnected2D(6, (5, 5), kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    UpSampling2D(size=(2, 2),interpolation='bilinear'),

    LocallyConnected2D(6, (3, 3), kernel_initializer='he_uniform'),
    LeakyReLU(),
    LocallyConnected2D(1, (2, 2), use_bias=False, kernel_initializer='glorot_normal'),
    Activation('relu')
])

latent = Input(shape=(latent_size, ))

image_class = Input(shape=(1, ), dtype='int32')
emb = Flatten()(Embedding(2, latent_size, input_length=1,
                          embeddings_initializer='glorot_normal')(image_class))

h = Multiply()([latent, emb])

fake_image = loc(h)

generator = Model(inputs=[latent, image_class], outputs=[fake_image])

generator.summary()




for batch_size_i in batch_size_i_array:


	batch_size = int(batch_size_i)
	batchsize = int(batch_size_i)

	print(' ')
	print(batch_size)


	@tf.function(experimental_relax_shapes=True)
	def body(loop_index, output):

		noise = tf.random.normal((batchsize, 200), 0, 1)
		class_i = tf.ones((batchsize, 1))

		logits = generator([noise, class_i], training=False)

		combined_output = tf.squeeze(logits,axis=-1)			

		return [loop_index+1, tf.concat([output, combined_output], axis=0)]

	print(loops_to_do/batchsize)


	print('warming up1')
	loop_index = tf.constant(0)

	output = tf.zeros([0, 25, 25], dtype=tf.float32)
	
	condition_func = lambda loop_index, output: loop_index < 1
	t0 = time.time()
	generated_training = tf.while_loop(condition_func, body, loop_vars=[loop_index, output])[1]
	t1 = time.time()
	total_time = t1-t0
	print('warm up1',np.shape(generated_training),'time',total_time)



	print('warming up')
	loop_index = tf.constant(0)

	output = tf.zeros([0, 25, 25], dtype=tf.float32)
	
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

	output = tf.zeros([0, 25, 25], dtype=tf.float32)

	
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


	with open("results_lagans.txt", "a") as myfile:
		myfile.write('%d, %.2f \n'%(batch_size, points_in_5_mins))



