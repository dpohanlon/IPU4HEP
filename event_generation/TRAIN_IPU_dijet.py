
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
print(tf.__version__)

from tensorflow.python.ipu import ipu_compiler,         \
                                  ipu_optimizer,        \
                                  ipu_estimator,        \
                                  scopes,               \
                                  loops,                \
                                  ipu_infeed_queue,     \
                                  ipu_outfeed_queue,    \
                                  utils,                \
                                  gradient_accumulation_optimizer as gao
from tensorflow.python.ipu.ops import normalization_ops




BATCH_SIZE=128
# load dataset

train_images = np.random.normal(0,1,(60000,7))

print(np.shape(train_images))


train_images = train_images.reshape(train_images.shape[0], 7).astype(
    "float32"
)


train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE,drop_remainder=True).repeat(10)
)


infeed_GAN = ipu_infeed_queue.IPUInfeedQueue(train_dataset, feed_name='in_GAN')

outfeed_FULL = ipu_outfeed_queue.IPUOutfeedQueue(feed_name='out_FULL')

outfeed_test = ipu_outfeed_queue.IPUOutfeedQueue(feed_name='out_test')


with tf.device("cpu"):
        numPoints = tf.placeholder(np.int32, shape=(), name="numPoints")

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


_EPSILON = K.epsilon()

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def _loss_generator(y_true, y_pred):
    y_pred = K.clip(y_pred, _EPSILON, 1.0-_EPSILON)
    out = -(K.log(y_pred))
    return K.mean(out, axis=-1)

GAN_noise_size = 128
GAN_output_size = 7
# Build Generator model ...
with tf.variable_scope('input'):
    G_input = Input(shape=(GAN_noise_size,))
with tf.variable_scope('gen'):
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
Generator = Model(G_input, G_output)
Generator.summary()

with tf.variable_scope('input'):
    # Build Discriminator model ...
    D_input = Input(shape=(GAN_output_size,))
with tf.variable_scope('dis'):
    D = Dense(128)(D_input)
    D = Reshape((8, 8, 2))(D)
    D = Conv2D(64, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(32, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Conv2D(16, kernel_size=3, strides=1, padding="same")(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Flatten()(D)
    D = LeakyReLU(alpha=0.2)(D)
    D = Dropout(0.2)(D)
    D_output = Dense(1, activation="sigmoid")(D)
Discriminator = Model(D_input, D_output)
Discriminator.summary()



optimizer_D = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
optimizer_stacked = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)



def train_model(X):
    d_vars = tf.trainable_variables('dis')
    g_vars = tf.trainable_variables('gen')  

    noise = tf.random.normal((BATCH_SIZE, 128), 0, 1)
    fake_images = Generator(noise)

    real_images = X

    in_values = tf.concat([fake_images, real_images],0)

    labels_D_0 = tf.zeros((BATCH_SIZE, 1)) 
    labels_D_1 = tf.ones((BATCH_SIZE, 1))

    labels_D = tf.concat([labels_D_0, labels_D_1],0)

    out_values = Discriminator(in_values)

    loss_D = tf.keras.losses.binary_crossentropy(labels_D,out_values)
    loss_D = tf.math.reduce_mean(loss_D)

    
    training_op_D = optimizer_D.minimize(loss_D, var_list = d_vars)

    noise_stacked = tf.random.normal((int(BATCH_SIZE*2), 128), 0, 1)
    fake_images2 = Generator(noise_stacked)
    stacked_output = Discriminator(fake_images2)

    labels_stacked = tf.ones((int(BATCH_SIZE*2), 1))

    loss_stacked = _loss_generator(labels_stacked,stacked_output)
    loss_stacked = tf.math.reduce_mean(loss_stacked)

    
    training_op_stacked = optimizer_stacked.minimize(loss_stacked, var_list=g_vars)

    return outfeed_FULL.enqueue([loss_D, loss_stacked, fake_images]), training_op_D, training_op_stacked

def test_model():

    noise = tf.random.normal((BATCH_SIZE, 128), 0, 1)
    fake_images = Generator(noise)

    return outfeed_test.enqueue(fake_images)


def training_loop_FULL(numPoints):
    out = loops.repeat(numPoints, train_model, infeed_queue=infeed_GAN)
    return out

def training_loop_test():
    out = loops.repeat(1, test_model)
    return out


# Compile the graph with the IPU custom xla compiler
with scopes.ipu_scope("/device:IPU:0"):
    compiled_FULL = ipu_compiler.compile(training_loop_FULL, [numPoints])

    compiled_test = ipu_compiler.compile(training_loop_test)


# Ops to read the outfeed and initialize all variables
dequeue_outfeed_op_FULL = outfeed_FULL.dequeue()
dequeue_outfeed_op_test = outfeed_test.dequeue()

init_op = tf.global_variables_initializer()


cfg = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = utils.auto_select_ipus(cfg, 1)
utils.configure_ipu_system(cfg)


t0 = time.time()

loss_list = np.empty((0,3))
# Run the model
with tf.Session() as sess:

    # Initialize
    sess.run(init_op)
    sess.run(infeed_GAN.initializer)
    # Run
    print('Running...')

    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for warm up',t1-t0)


    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1000})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for 1000',t1-t0)



    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1000})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for 1000',t1-t0)


    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1000})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for 1000',t1-t0)







