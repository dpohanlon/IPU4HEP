import numpy as np
import argparse

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Embedding, Multiply, Activation, Conv2D, ZeroPadding2D, LocallyConnected2D, Concatenate, GRU, Lambda, UpSampling1D, Conv2DTranspose
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


''' Create the network. '''

print('batch_size:',batch_size)

GAN_noise_size = 128
GAN_output_size = 7
G_input = Input(shape=(GAN_noise_size,))

G = Dense(128, kernel_initializer='glorot_uniform')(G_input)
#G = Dropout(0.2)(G)
G = LeakyReLU(alpha=0.2)(G)
#G = Activation("relu")(G)
G = BatchNormalization()(G)

G = Reshape([8, 8, 2])(G)  # default: channel last

G = Conv2DTranspose(32, kernel_size=2, strides=1, padding="same")(G)
#G = Activation("relu")(G)
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



outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(feed_name="outfeed%d"%np.random.randint(low=0,high=99999))


with tf.device("cpu"):
	numPoints = tf.placeholder(np.int32, shape=(), name="numPoints")


def body():

	noise = tf.random.normal((batch_size, 128), 0, 1)

	logits = generator([noise], training=False)

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

total_times = np.empty(0)

with tf.Session() as sess:

	# Warm-up run to compile the graph
	t0_compile = time.time() 
	print("Warming Up...")
	sess.run(tf.global_variables_initializer())

	sess.run(run_loop, feed_dict={numPoints: 1}) # just one point here
	out_ = sess.run(dequeue_outfeed)
	t1_compile = time.time() 

	# The real run

	for i in range(0, 1):
		t0 = time.time() # Should show actual run time now, not run+compile time
		sess.run(run_loop, feed_dict={numPoints: int(loops_to_do/batch_size)}) # 1E5 events for test
		# sess.run(run_loop, feed_dict={numPoints: loops_to_do}) # 1E5 events for test
		t1 = time.time()


		# Get our result from the outfeed queue
		generated_training = sess.run(dequeue_outfeed)

		''' If run with a batchsize more than 1, reshape the output. '''
		if batch_size > 1:
			generated_training = np.reshape(generated_training, (np.shape(generated_training)[0]*np.shape(generated_training)[1],7))
		print(np.shape(generated_training))
		total_times = np.append(total_times, t1-t0)


print(total_times)


for total_time in total_times:
	# total_time = t1-t0
	total_time_compile = t1_compile-t0_compile

	print(generated_training.shape)

	points_per_second = (np.shape(generated_training)[0]/total_time)

	print('compile time:',int(total_time_compile))

	print('Time to generate %d points: %.3fs'%(np.shape(generated_training)[0],total_time))

	print('Points generated in 1 sec: %d.'%int(points_per_second),'batch_size',batch_size)

	''' Save results '''
	with open("results_dijet_3.txt", "a") as myfile:
		myfile.write('%d, %.2f \n'%(batch_size, points_per_second))








