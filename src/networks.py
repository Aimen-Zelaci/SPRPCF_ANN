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
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from tensorflow.python.client import device_lib

# Print CPUs / GPUs available
print(device_lib.list_local_devices())
# Enable Tensorflow eager execution. This line must be at the top of the script.
tf.compat.v1.enable_eager_execution()

############################### ANN MODEL ##########################################
####################################################################################
####################################################################################

def make_model(flags):
    model = tf.keras.Sequential()
    num_inputs = flags.num_inputs
    num_neurons = flags.num_neurons
    num_layers = flags.num_layers
    num_outputs = flags.num_outputs
    batch_norm = flags.batch_norm
    # 1st layer
    model.add(layers.Dense(num_neurons, input_shape=(num_inputs,)))
    model.add(layers.ReLU())
    # Hidden layers
    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_neurons))
        if batch_norm == "True":
            model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
    # OUTPUT layer
    model.add(layers.Dense(num_outputs))
    model.add(layers.ReLU())

    return model

def train_model(model ,tr_data, tr_labels, va_data, va_labels, flags):
    start = time.time()
    print('\n\n Training the ANN model on {} samples \n\n'.format(int(len(tr_data))))
    epochs = flags.epochs
    lr = flags.lr
    batch_size = flags.ANN_batch_size
    save_dir = flags.save_dir
    chkdir = flags.chkdir
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
    # Define optimizers
    Adam = keras.optimizers.Adam(learning_rate=
        lr, beta_1=0.9, beta_2=0.999)
    RMSprop = keras.optimizers.RMSprop(learning_rate=1e-2, rho=0.9)
    # Train
    model.compile(optimizer=Adam, loss='mean_squared_error')
    hist = model.fit(tr_data, tr_labels, epochs=epochs, verbose=1,
                     validation_data=(va_data, va_labels), callbacks=[checkpointer, tensorboard_callback], batch_size=batch_size)
    # Post training
    model.save(save_dir)
    #loss = hist.history['loss']
    print('\n*****\nTraining run time for data set length {} is : {} sec\n*****'.format(336+flags.augment_size, time.time() - start))

def load_model(model,load_type,dir):
    if load_type == 'load_weights':
        model.load_weights(dir)
    if load_type == 'load_model':
        model = keras.models.load_model(dir)

    return model

def test_model(model, test_data, test_labels, len_tr_data=0, plot=True):
    start = time.time()
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
            wavelength_smoothed = sp.linspace(500,820,100)
            spl.set_smoothing_factor(0.0)
            plt.plot(wavelength_smoothed, spl(wavelength_smoothed), label='Predictions', c='black', lw=2)
            plt.scatter(wavelength, test_labels[i:i+16], label='Actual', c='red', alpha=0.7,marker='o')
            legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
            legend.get_frame().set_facecolor('C7')
            plt.ylabel('Confinment loss in Log10(db/cm)')
            plt.xlabel('Wavelegth in nm')
            plt.grid()
            plt.title('Length of TR dataset 336 + {}'.format(len_tr_data))
            plt.show()
    print('\n*****\nTest run time of the ANN model is: {} sec\n*****\n'.format(time.time() - start))
    return predictions

    ############################### WGAN ###############################################
    ####################################################################################
    ####################################################################################

class Wgan(object):

    def __init__(self,flags):
        self.flags = flags
        self.BATCH_SIZE = flags.wgan_batch_size
        self.noise_dim = flags.noise_dim
        self.num_critic_input = flags.num_critic_input
        self.n_critic = flags.n_critic
        self.grad_penalty_weight = flags.grad_penalty_weight
        self.num_examples_to_generate = flags.num_examples_to_generate
        self.epochs = flags.wgan_epochs
        self.critic_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, beta_2=0.9)
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self._tensorboard()

    def make_generator_model(self):
        num_layers = self.flags.gen_num_layers
        batch_norm = self.flags.gen_batch_norm
        model = tf.keras.Sequential()
        # 1st layer
        model.add(layers.Dense(self.BATCH_SIZE * (2 ** 2), input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # HIDDEN layers
        for _ in range(num_layers - 1):
            model.add(layers.Dense(self.BATCH_SIZE * (2 ** 2)))
            if batch_norm == "True":
                model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
        # OUTPUT layer
        model.add(layers.Dense(self.noise_dim))
        model.add(layers.ReLU())

        return model

    def make_critic_model(self):
        num_layers = self.flags.cr_num_layers
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
            if i < 0:
                i = 0
        # OUTPUT layer
        model.add(layers.Dense(1, activation='linear'))

        return model

    def gradient_penalty(self,model,x_real, x_fake):
        alpha = tf.compat.v1.random_uniform(shape=[self.BATCH_SIZE, 1], minval=0., maxval=1.)
        diff = x_fake - x_real
        interpolates = x_real + (alpha * diff)
        gradients = tf.gradients(model(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.compat.v1.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    @tf.function
    def train_critic(self,data, generator, critic, epoch):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        with tf.GradientTape() as cr_tape, self.critic_summary_writer.as_default():
            generated_data = generator(noise, training=True)
            data = tf.dtypes.cast(data, tf.float32)
            real_output = critic(data, training=True)
            fake_output = critic(generated_data, training=True)
            critic_loss = self.critic_loss(real_output, fake_output)
            gp =self. gradient_penalty(critic, data, generated_data)
            critic_loss += self.grad_penalty_weight * gp
            #self.train_loss_cr(critic_loss)
            #tf.summary.scalar('loss', self.train_loss_cr.result(), step=epoch)
        gradients_of_critic = cr_tape.gradient(critic_loss, critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    @tf.function
    def train_gen(self, generator, critic, epoch):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        with tf.GradientTape() as gen_tape, self.gen_summary_writer.as_default():
            generated_data = generator(noise, training=True)
            fake_output = critic(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output)
            #self.train_loss_gen(gen_loss)
            #tf.summary.scalar('loss', self.train_loss_gen.result(), step=epoch)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    def shape_wgan_data(self,tr_data, tr_labels):
        BUFFER_SIZE = int(len(tr_data))
        # Place the labels and inputs in one vector
        dataset = sp.concatenate((tr_data, tr_labels), axis=1)
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(self.BATCH_SIZE)

        return train_dataset

    def _tensorboard(self):
        # Critic Tensorboard loss summary
        critic_logdir = r'.\ganlogs\scalars\{}'.format(time.time())
        train_loss_cr = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        critic_summary_writer = tf.compat.v2.summary.create_file_writer(critic_logdir)
        # Generator TensorBoard loss summary
        gen_logdir = r'.\ganlogs\scalars\{}'.format(time.time())
        train_loss_gen = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        gen_summary_writer = tf.compat.v2.summary.create_file_writer(gen_logdir)

        self.gen_summary_writer = gen_summary_writer
        self.critic_summary_writer = critic_summary_writer
        self.train_loss_gen = train_loss_gen
        self.train_loss_cr = train_loss_cr

    def train_wgan(self,tr_data, tr_labels, generator, critic):
        start_tr = time.time()
        print('\n*****\nTraining The WGAN  \n****\n')
        epochs = self.epochs
        # Continue training
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        # Save model's checkpoints as we iterate
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         generator=generator,
                                         discriminator=critic)
        dataset = self.shape_wgan_data(tr_data, tr_labels)
        for epoch in range(epochs):
            start = time.time()
            for data_batch in dataset:
                # Train Critic
                for _ in range(self.n_critic):
                    self.train_critic(data_batch, generator, critic, epoch)
                # Train Generator
                self.train_gen(generator, critic, epoch)
            # Generate and save data
            # generate_and_save_data(generator,seed,training=False)
            # Plot generated data vs wavelength each 200 epoch
            if (epoch + 1) % 2 == 0:
                # data_handler.plot_wgan(epoch + 1)
                checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        print('\n*****\nTraining run time for the wgan is: {} sec\n****\n'.format(time.time() - start_tr))

    def generate_and_save_data(self,iterations,training, generator, critic):
        print('\n*****\n GENERATING DATA ... \n****\n')

        start = time.time()
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         generator=generator,
                                         discriminator=critic)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        _df = sp.zeros(7).reshape(1,7)
        for _ in tqdm(range(iterations)):
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

        print('\n*****\nGeneration run time is: {} sec\n*****'.format(time.time() - start))


    @staticmethod
    def critic_loss(real_output, fake_output):
        return -(tf.reduce_mean(real_output) - tf.reduce_mean(fake_output))

    @staticmethod
    def generator_loss(fake_output):
        return -tf.reduce_mean(fake_output)
