# Load libraries
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from scipy.interpolate import UnivariateSpline
from tensorflow.python.client import device_lib

# Print CPUs / GPUs available
print(device_lib.list_local_devices())
# Enable Tensorflow eager execution. This line must be at the top of the script.
tf.compat.v1.enable_eager_execution()

############################### ANN MODEL ##########################################
####################################################################################
####################################################################################

def make_model(num_layers=6, num_neurons=50, num_inputs=6, num_outputs=1):
    model = tf.keras.Sequential()

    # 1st layer
    model.add(layers.Dense(num_neurons, input_shape=(num_inputs,)))
    model.add(layers.ReLU())

    # Hidden layers
    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_neurons))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    # OUTPUT layer
    model.add(layers.Dense(num_outputs))
    model.add(layers.ReLU())

    return model

def train_model(model,epochs, tr_data, tr_labels, va_data, va_labels, save_dir, chkdir):

    # Proceed training of a saved model
    # model = keras.models.load_model(save_dir)

    # TensorBoard log directory
    logdir = r'.\logs\scalars\{}'.format(time.time())

    # CALLBACKS
    # TensorBoard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Save the best va_loss file
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=chkdir, verbose=0,
                                                   save_best_only=True)

    data_size = int(len(tr_data))
    # Learning rate / Minibatch size
    lr = 1e-4
    batch_size = 8
    if data_size >= 2000:
        lr = 2e-4
        batch_size = 16
    if data_size >= 3000:
        lr = 2.5e-4
        batch_size = 20

    # Define optimizers
    Adam = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    RMSprop = keras.optimizers.RMSprop(learning_rate=1e-2, rho=0.9)
    # Train
    model.compile(optimizer=Adam, loss='mean_squared_error')
    hist = model.fit(tr_data, tr_labels, epochs=epochs, verbose=1,
                     validation_data=(va_data, va_labels), callbacks=[checkpointer], batch_size=batch_size)
    # Post training
    model.save(save_dir)
    #loss = hist.history['loss']
    #epochsArr = np.arange(epochs)
    #predictions = model.predict([va_data])
    #MSE = sp.square(sp.subtract(va_labels, predictions)).mean()

def load_model(model,load_type,dir):
    if load_type == 'load_weights':
        model.load_weights(dir)

    if load_type == 'load_model':
        model = keras.models.load_model(dir)

    return model

def test_model(model, test_data, test_labels, len_tr_data=0, plot=True):
    predictions = model.predict([test_data])
    #MSE = sp.square(sp.subtract(test_labels, predictions)).mean()
    if plot == True:
        plt.scatter(test_labels, predictions, label='Predictions', c='red', alpha=0.5)
        plt.plot(test_labels, test_labels, label='Actual', c='green', alpha=0.7)
        legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
        legend.get_frame().set_facecolor('C7')
        plt.ylabel('Prediction Log10(loss)')
        plt.xlabel('Actual Log10(loss)')
        plt.grid()
        plt.title('Length of TR dataset 336 + {}'.format(len_tr_data))
        plt.show()

        wavelength = sp.arange(500,820,20)
        for i in [0,16,32]:
            spl = UnivariateSpline(wavelength, predictions[i:i+16])
            wavelength_smoothed = sp.linspace(500,820,500)
            pr_smoothed = spl(wavelength_smoothed)
            spl.set_smoothing_factor(0.5)
            plt.plot(wavelength_smoothed, pr_smoothed, label='Predictions', c='black', lw=2)
            plt.scatter(wavelength, test_labels[i:i+16], label='Actual', c='red', alpha=0.7,marker='o')
            legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
            legend.get_frame().set_facecolor('C7')
            plt.ylabel('Confinment loss in Log10(db/cm)')
            plt.xlabel('Wavelegth in nm')
            plt.grid()
            plt.title('Length of TR dataset 336 + {}'.format(len_tr_data))
            plt.show()

    return predictions



    ############################### WGAN ###############################################
    ####################################################################################
    ####################################################################################

class Wgan(object):

    def __init__(self,BATCH_SIZE = 12,
                      noise_dim = 7,
                      num_critic_input = 7,
                      n_critic = 5,
                      grad_penalty_weight = 10,
                      num_examples_to_generate = 8,
                      critic_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9),
                      generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)
                        ):

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.num_critic_input = num_critic_input
        self.n_critic = n_critic
        self.grad_penalty_weight = grad_penalty_weight
        self.num_examples_to_generate = num_examples_to_generate
        self.critic_optimizer = critic_optimizer
        self.generator_optimizer = generator_optimizer

    def make_generator_model(self, num_layers = 5):
        model = tf.keras.Sequential()

        # 1st layer
        model.add(layers.Dense(self.BATCH_SIZE * (2 ** 2), input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        # HIDDEN layers
        for _ in range(num_layers - 1):
            model.add(layers.Dense(self.BATCH_SIZE * (2 ** 2)))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

        # OUTPUT layer
        model.add(layers.Dense(self.noise_dim))
        model.add(layers.ReLU())

        return model

    def make_critic_model(self, num_layers = 5):
        model = tf.keras.Sequential()

        # 1st layer
        model.add(layers.Dense(self.BATCH_SIZE * (2 ** 2), input_shape=(self.num_critic_input,)))
        model.add(layers.LeakyReLU())

        # HIDDEN LAYERS
        i = 2
        for _ in range(num_layers - 1):
            model.add(layers.Dense(self.BATCH_SIZE * (2 ** i)))
            model.add(layers.LeakyReLU())
            i-=1
            if i == 0:
                i = 0
        # OUTPUT layer
        model.add(layers.Dense(1, activation='linear'))

        return model

    def gradient_penalty(self,model,x_real, x_fake):
        alpha = tf.compat.v1.random_uniform(shape=[self.BATCH_SIZE, self.num_critic_input], minval=0., maxval=1.)
        diff = x_real - x_fake
        interpolates = x_real + (alpha * diff)
        gradients = tf.gradients(model(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    @tf.function
    def train_critic(self,data, generator, critic, train_critic_loss_summary):

        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as disc_tape:
            generated_data = generator(noise, training=True)
            data = tf.dtypes.cast(data, tf.float32)
            real_output = critic(data, training=True)
            fake_output = critic(generated_data, training=True)

            # critic loss
            disc_loss = self.critic_loss(real_output, fake_output)

            # Gradient penalty
            gp =self. gradient_penalty(critic, data, generated_data)
            disc_loss += self.grad_penalty_weight * gp
            train_critic_loss_summary(disc_loss)
        gradients_of_critic = disc_tape.gradient(disc_loss, critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    @tf.function
    def train_gen(self, generator, critic, train_loss_gen_summary):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        with tf.GradientTape() as gen_tape:
            generated_data = generator(noise, training=True)
            fake_output = critic(generated_data, training=True)
            # WGAN
            # generator loss
            gen_loss = self.generator_loss(fake_output)
            train_loss_gen_summary(gen_loss)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    def shape_wgan_data(self,tr_data, tr_labels):
        BUFFER_SIZE = int(len(tr_data))
        # Place the labels and inputs in one vector
        dataset = sp.concatenate((tr_data, tr_labels), axis=1)
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(self.BATCH_SIZE)

        return train_dataset

    def train_wgan(self,tr_data, tr_labels, epochs, generator, critic):
        # Continue training
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        # Save model's checkpoints as we iterate
        checkpoint_dir = './training_checkpoints'
        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         generator=generator,
                                         discriminator=critic)

        # Critic Tensorboard loss summary
        critic_logdir = r'.\ganlogs\scalars\{}'.format(time.time())
        train_loss_cr = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        critic_summary_writer = tf.compat.v2.summary.create_file_writer(critic_logdir)
        # Generator TensorBoard loss summary
        gen_logdir = r'.\ganlogs\scalars\{}'.format(time.time())
        train_loss_gen = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        gen_summary_writer = tf.compat.v2.summary.create_file_writer(gen_logdir)

        dataset = self.shape_wgan_data(tr_data, tr_labels)
        for epoch in range(epochs):
            start = time.time()

            for data_batch in dataset:
                # Train Critic
                for _ in range(self.n_critic):
                    self.train_critic(data_batch, generator, critic, train_loss_cr)
                    with critic_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss_cr.result(), step=epoch)
                # Train Generator
                self.train_gen(generator, critic, train_loss_gen)
                with gen_summary_writer.as_default():
                    tf.summary.scalar('loss', train_loss_gen.result(), step=epoch)

            # Generate and save data
            # generate_and_save_data(generator,seed,training=False)
            # Plot generated data vs wavelength each 200 epoch
            if (epoch + 1) % 2 == 0:
                # data_handler.plot_wgan(epoch + 1)
                checkpoint.save(file_prefix=checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    def generate_and_save_data(self,iterations,training, generator, critic):
        checkpoint_dir = './training_checkpoints'
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         generator=generator,
                                         discriminator=critic)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        _df = sp.zeros(7).reshape(1,7)

        for _ in range(iterations):
            seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
            predictions = generator(seed, training=training)
            if (training == False):
                for p in predictions:
                    # Noise filter
                    ones = sum(int(o >= 1.0) for o in p[:6])
                    zeros = sum(int(z < 0.1) for z in p)
                    # sp.savetxt(r'gen_data_pcf.txt', p, delimiter=',')
                    if(ones == 0 and zeros == 0):
                        _df = np.concatenate((_df,tf.reshape(p,[1, 7])), axis=0)
        _df = pd.DataFrame(_df, index=None)
        _df = _df.drop(0, axis=0)
        _df.to_csv('.\gen_data\gen_data2.txt', index=False)

    @staticmethod
    def critic_loss(real_output, fake_output):
        return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))

    @staticmethod
    def generator_loss(fake_output):
        return -tf.reduce_mean(fake_output)
