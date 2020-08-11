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

from tensorflow.python import ipu
from tensorflow.python.ipu import loops, ipu_outfeed_queue
from tensorflow.python.ipu.scopes import ipu_scope
cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

print(tf.__version__)


loops_to_do = int(1E6)


parser = argparse.ArgumentParser()

parser.add_argument('-b', action='store', dest='batchsize', type=int, default=25)

results = parser.parse_args()

batch_size = results.batchsize


#################################################
''' Create the network. '''
factor = 1

print('FACTOR:',factor)
''' Create the network. '''
latent_size = 200

loc = Sequential([
    Dense(int(factor*128) * 7 * 7, input_dim=latent_size),
    Reshape((7, 7, int(factor*128))),

    Conv2D(int(factor*64), (5, 5), padding='same', kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    UpSampling2D(size=(2, 2),interpolation='bilinear'),

    ZeroPadding2D((2, 2)),
    # LocallyConnected2D(int(factor*6), (5, 5), kernel_initializer='he_uniform'),
    Conv2D(int(factor*6), (5, 5), kernel_initializer='he_uniform'),
    LeakyReLU(),
    BatchNormalization(),
    UpSampling2D(size=(2, 2),interpolation='bilinear'),

    # LocallyConnected2D(int(factor*6), (3, 3), kernel_initializer='he_uniform'),
    Conv2D(int(factor*6), (3, 3), kernel_initializer='he_uniform'),
    LeakyReLU(),
    # LocallyConnected2D(1, (2, 2), use_bias=False, kernel_initializer='glorot_normal'),
    Conv2D(1, (2, 2), use_bias=False, kernel_initializer='glorot_normal'),
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




outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed%d"%np.random.randint(low=0,high=99999))


with tf.device("cpu"):
	numPoints = tf.placeholder(np.int32, shape=(), name="numPoints")


def body():

	noise = tf.random.normal((batch_size, 200), 0, 1)
	class_i = tf.ones((batch_size, 1))

	logits = generator([noise, class_i], training=False)

	images = tf.squeeze(logits)

	outfeed = outfeed_queue.enqueue(images)
	return outfeed

# Function which we will combine -- runs our inference inside
# a loop, so not passing a massive (N, M) tensor around
def getImages(numPoints):
	r = loops.repeat(numPoints, body, [])
	return r


# Here we compile the graph
with ipu_scope("/device:IPU:0"):
	run_loop = ipu.ipu_compiler.compile(getImages, [numPoints])

# Function to get the output back
dequeue_outfeed = outfeed_queue.dequeue()

with tf.Session() as sess:

	# Warm-up run to compile the graph
	t0_compile = time.time() 
	print("Warming Up...")
	sess.run(tf.global_variables_initializer())

	sess.run(run_loop, feed_dict={numPoints: 1}) # just one point here
	out_ = sess.run(dequeue_outfeed)
	t1_compile = time.time() 

	# The real run
	t0 = time.time() # Should show actual run time now, not run+compile time
	sess.run(run_loop, feed_dict={numPoints: int(loops_to_do/batch_size)}) # 1E5 events for test
	# sess.run(run_loop, feed_dict={numPoints: loops_to_do}) # 1E5 events for test
	t1 = time.time()


	# Get our result from the outfeed queue
	generated_training = sess.run(dequeue_outfeed)

	''' If run with a batchsize more than 1, reshape the output. '''
	if batch_size > 1:
		generated_training = np.reshape(generated_training, (np.shape(generated_training)[0]*np.shape(generated_training)[1],25,25))


total_time = t1-t0
total_time_compile = t1_compile-t0_compile

print(generated_training.shape)

points_per_second = (np.shape(generated_training)[0]/total_time)

print('compile time:',int(total_time_compile))

print('Time to generate %d points: %.3fs'%(np.shape(generated_training)[0],total_time))

print('Points generated in 1 sec: %d.'%int(points_per_second),'batch_size',batch_size)

''' Save results '''
with open("results_lagans.txt", "a") as myfile:
	myfile.write('%d, %.2f \n'%(batch_size, points_per_second))

