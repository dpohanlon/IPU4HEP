import numpy as np
import argparse

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, Embedding, Multiply, Activation, Conv2D, ZeroPadding2D, LocallyConnected2D, Concatenate, GRU, Lambda
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import UpSampling2D

import tensorflow as tf
# tf.disable_v2_behavior()

import time
from tensorflow.keras import backend as K

_EPSILON = K.epsilon()
def _loss_generator(y_true, y_pred):
	y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
	out = -(K.log(y_pred))
	return K.mean(out, axis=-1)


print(tf.__version__)

train_images = np.random.normal(0,1,(60000,4))

print(np.shape(train_images))

batch_size = 50

train_images = train_images.reshape(train_images.shape[0], 1, 4).astype(
	"float32"
)


train_dataset = (
	tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size,drop_remainder=True).repeat(1)
)


# Build Generator model ...
X = Input(shape=(1,100))
H = Dense(512)(X)
H = LeakyReLU(alpha=0.2)(H)
H = Dense(1024)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dense(4, activation='tanh')(H)
H = Reshape((1,4))(H)
Generator = Model(X, H)
Generator.summary()


# Build Discriminator model ...
X = Input(shape=(1,4))
H = Flatten()(X)
H = Dense(768)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dense(1536)(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dense(1, activation='sigmoid')(H)
Discriminator = Model(X, H)
Discriminator.summary()

optimizer_D = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
optimizer_stacked = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)


batch_size = 50

gen_optimizer = tf.keras.optimizers.Adam(0.0002)

disc_optimizer = tf.keras.optimizers.Adam(0.0002)


@tf.function
def train_step(images):
	noise = tf.random.normal([batch_size, 1, 100])
	noise_stacked = tf.random.normal((int(batch_size*2), 1, 100), 0, 1)

	with tf.GradientTape(persistent=True) as tape:
		generated_images = Generator(noise)

		in_values = tf.concat([generated_images, images],0)
		labels_D_0 = tf.zeros((batch_size, 1)) 
		labels_D_1 = tf.ones((batch_size, 1))

		labels_D = tf.concat([labels_D_0, labels_D_1],0)

		out_values = Discriminator(in_values)

		loss_D = tf.keras.losses.binary_crossentropy(labels_D,out_values)
		loss_D = tf.math.reduce_mean(loss_D)


		fake_images2 = Generator(noise_stacked)
		stacked_output = Discriminator(fake_images2)

		labels_stacked = tf.ones((int(batch_size*2), 1))

		loss_stacked = _loss_generator(labels_stacked,stacked_output)
		loss_stacked = tf.math.reduce_mean(loss_stacked)
		
	grad_disc = tape.gradient(loss_D, Discriminator.trainable_variables)
	grad_gen = tape.gradient(loss_stacked, Generator.trainable_variables)

	disc_optimizer.apply_gradients(zip(grad_disc, Discriminator.trainable_variables))
	gen_optimizer.apply_gradients(zip(grad_gen, Generator.trainable_variables))

	return loss_stacked, loss_D


iteration = 0 
for epoch in range(10):
	start = time.time()
	total_gen_loss = 0
	total_disc_loss = 0

	for images in train_dataset:
		iteration += 1

		if iteration == 200:
			t0 = time.time()
		if iteration == 1200:
			t1 = time.time()

			print('1000',t1-t0)
			t0 = time.time()
		if iteration == 2200:
			t1 = time.time()

			print('2000',t1-t0)
			t0 = time.time()

		if iteration == 3200:
			t1 = time.time()

			print('3000',t1-t0)
			t0 = time.time()
			quit()
		gen_loss, disc_loss = train_step(images)


