# Load libraries
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
import os
import data_loader

# Enable Tensorflow eager execution. This line must be at the top of the script.
tf.compat.v1.enable_eager_execution()


############################### ANN MODEL ##########################################
####################################################################################
####################################################################################

def make_model():
    model = tf.keras.Sequential()

    # INPUT layer
    model.add(layers.Dense(50, input_shape=(6,)))
    model.add(layers.ReLU())

    # Hidden layers
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # OUTPUT layer
    model.add(layers.Dense(1))
    model.add(layers.ReLU())

    return model


def train_model(epochs, tr_data, tr_labels, va_data, va_labels, save_dir, chkdir):
    # Create new model
    model = make_model()

    # Proceed training of a saved model
    # model = keras.models.load_model(save_dir)

    # TensorBoard log directory
    logdir = r'\logs\1000\scalars\{}'.format(time())

    # CALLBACKS
    # TensorBoard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Save the best va_loss file
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=chkdir, verbose=1,
                                                   save_best_only=True)
    # Train
    data_size = int(len(tr_data))
    # Learning rate / Minibatch size
    lr = 1e-4
    batch_size = 8

    if data_size >= 2000:
        lr = 2e-4
        batch_size = 16

    if data_size >= 3000:
        lr = 20e-4 / 8
        batch_size = 20

    # Define optimizers
    Adam = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    RMSprop = keras.optimizers.RMSprop(learning_rate=1e-2, rho=0.9)

    model.compile(optimizer=Adam, loss='mean_squared_error', batch_size=batch_size)
    hist = model.fit(tr_data, tr_labels, epochs=epochs, verbose=2,
                     validation_data=(va_data, va_labels), callbacks=[tensorboard_callback, checkpointer])

    # Post training
    model.save(save_dir)
    loss = hist.history['loss']
    epochsArr = sp.arange(epochs)
    predictions = model.predict([va_data])
    MSE = sp.square(sp.subtract(va_labels, predictions)).mean()

    ### Plot predictions vs validation ########
    plt.scatter(va_labels, predictions)  #
    plt.plot(va_labels, va_labels, 'r')  #
    plt.grid()  #
    plt.title("epochs {}; "  #
              "length of dataset {}\n ; "  # 
              "TR LOSS {}\n; Test MSE {}"  #
              .format(epochs,  #
                      int(len(tr_data)),  #
                      loss[-1], MSE))  #
    plt.show()  #
    ### TR loss vs epochs #####################
    plt.plot(epochsArr, loss, 'r')  #
    plt.grid()  #
    plt.show()  #
    ############# Graphs ######################
    ###########################################


def load_model(dir):
    if ('load_weights'):
        model = make_model()
        model.load_weights(dir)

    if ('load_model'):
        model = keras.models.load_model(dir)

    return model

    ############################### WGAN ###############################################
    ####################################################################################
    ####################################################################################


n_critic = 5
grad_penalty_weight = 10
BATCH_SIZE = 12
EPOCHS = 2000
noise_dim = 7
num_examples_to_generate = 8

def make_generator_model():
    model = tf.keras.Sequential()

    # INPUT layer
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2), input_shape=(7,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # Hidden layers
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # OUTPUT layer
    model.add(layers.Dense(7))
    model.add(layers.ReLU())

    return model


def make_critic_model():
    model = tf.keras.Sequential()

    # INPUT layer
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2), input_shape=(7,)))
    model.add(layers.LeakyReLU())

    # Hidden layers
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(BATCH_SIZE * (2 ** 2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(BATCH_SIZE * (2 ** 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(BATCH_SIZE * (2 ** 0)))
    model.add(layers.LeakyReLU())

    # OUTPUT layer
    model.add(layers.Dense(1, activation='linear'))

    return model


critic = make_critic_model()
generator = make_generator_model()


def critic_loss(real_output, fake_output):
    return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))


def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)


def gradient_penalty(x_real, x_fake):
    alpha = tf.compat.v1.random_uniform(shape=[BATCH_SIZE, 7], minval=0., maxval=1.)
    diff = x_real - x_fake
    interpolates = x_real + (alpha * diff)
    gradients = tf.gradients(critic(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp


# WGAN optimizers
critic_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)

# Save model's checkpoints as we iterate
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=critic_optimizer,
                                 generator=generator,
                                 discriminator=critic)

# Critic Tensorboard loss summary
critic_logdir = r"\ganlogs\scalars\{}".format(time.time())
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
critic_summary_writer = tf.compat.v2.summary.create_file_writer(critic_logdir)

# Generator TensorBoard loss summary
gen_logdir = r"\ganlogs\scalars\{}".format(time.time())
train_loss_gen = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
gen_summary_writer = tf.compat.v2.summary.create_file_writer(gen_logdir)


@tf.function
def train_critic(data):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)
        data = tf.dtypes.cast(data, tf.float32)
        real_output = critic(data, training=True)
        fake_output = critic(generated_data, training=True)
        # WGAN
        # critic loss
        disc_loss = critic_loss(real_output, fake_output)

        # Gradient penalty
        gp = gradient_penalty(data, generated_data)
        disc_loss += grad_penalty_weight * gp
        train_loss(disc_loss)
    gradients_of_critic = disc_tape.gradient(disc_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))


@tf.function
def train_gen():
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape:
        generated_data = generator(noise, training=True)
        fake_output = critic(generated_data, training=True)
        # WGAN
        # generator loss
        gen_loss = generator_loss(fake_output)
        train_loss_gen(gen_loss)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def generate_and_save_data(model, training):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    predictions = model(seed, training=False)
    if (training == False):
        for p in predictions:

            # Noise filter
            ones = sum(int(o >= 1.0) for o in p[:6])
            zeros = sum(int(z < 0.1) for z in p)
            if (zeros == 0 and ones == 0):
                f = open(r'\gen_data\data.tsv', 'a+')
                f.write("{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}\n".format(p[0], p[1], p[2], p[3], p[4], p[5], p[6]))
                f.close()


def shape_wgan_data(tr_data, tr_labels):
    BUFFER_SIZE = int(len(tr_data))
    dataset = []

    # Place the labels and inputs in one vector
    for x, y in zip(tr_data, tr_labels):
        temp = sp.array(sp.zeros(7))
        for i in range(6):
            temp[i] = x[i]
        temp[-1] = y
        dataset.append(temp)

    tr_data = sp.array([x for x in dataset])
    train_dataset = tf.data.Dataset.from_tensor_slices(tr_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset


def train_wgan(tr_data, tr_labels, epochs):
    dataset = shape_wgan_data(tr_data, tr_labels)
    for epoch in range(epochs):
        start = time.time()

        for data_batch in dataset:
            # Train Critic
            for _ in range(n_critic):
                train_critic(data_batch)
                with critic_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss.result(), step=epoch)
            # Train Generator
            train_gen()
            with gen_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss_gen.result(), step=epoch)

        # Generate and save data
        # generate_and_save_data(generator,seed,training=False)
        # Plot generated data vs wavelength each 200 epoch
        if (epoch + 1) % 200 == 0:
            data_loader.plot_wgan(epoch + 1)
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
