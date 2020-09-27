from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow.compat.v1 as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import time
import logging
from tqdm import tqdm
from scipy.interpolate import UnivariateSpline
from tensorflow.python.client import device_lib

# Print CPUs / GPUs available
print(device_lib.list_local_devices())
# disable Tensorflow eager execution. This line must be at the top of the script.
tf.disable_eager_execution()

# init logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


############################### ANN MODEL ##########################################


def make_model(flags):
    model = tf.keras.Sequential()
    num_inputs = flags.num_inputs
    num_neurons = flags.num_neurons
    num_layers = flags.num_layers
    num_outputs = flags.num_outputs
    batch_norm = flags.batch_norm
    logger.info('\n[*] Number of inputs: {}'.format(num_inputs))
    logger.info('\n[*] Number of hidden nodes: {}'.format(num_neurons))
    logger.info('\n[*] Number of hidden layers: {}'.format(num_layers))
    logger.info('\n[*] Number of outputs: {}'.format(num_outputs))
    logger.info('\n[*] Batch Normalization: {}'.format(batch_norm))
    # 1st layer
    model.add(layers.Dense(num_neurons,input_shape=(num_inputs,)))
    if batch_norm:
        pass
        #model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    # Hidden layers
    for _ in range(num_layers - 1):
        model.add(layers.Dense(num_neurons))
        if batch_norm:
            #model.add(layers.Dropout(0.5))
            model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
    # OUTPUT layer
    model.add(layers.Dense(num_outputs))
    model.add(layers.ReLU())

    return model


def train_model(model, tr_data, tr_labels, va_data, va_labels, flags):
    start = time.time()
    print('\n\n Training the ANN model on {} samples \n\n'.format(int(len(tr_data))))
    epochs = flags.epochs
    lr = flags.lr
    batch_size = flags.ANN_batch_size
    if flags.k_fold:
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
        if not os.path.isdir('./trained-kf-models/{}'.format(cur_time)):
            os.makedirs('./trained-kf-models/{}'.format(cur_time))

        if not os.path.isdir('./trained-kf-weights/{}'.format(cur_time)):
            os.makedirs('./trained-kf-weights/{}'.format(cur_time))

        save_dir = "./trained-kf-models/{}/model.h5".format(cur_time)
        chkdir = "./trained-kf-weights/{}/weights.hdf5".format(cur_time)
        flags.model_to_test = save_dir

    if not flags.k_fold:
        save_dir = flags.save_dir
        chkdir = flags.chkdir

    logger.info('\n[*] learning rate: {}'.format(lr))
    logger.info('\n[*] Batch size: {}'.format(batch_size))
    logger.info('\n[*] Epoch: {}'.format(epochs))
    # Proceed training of a saved model
    # model = keras.models.load_model(save_dir)
    # TensorBoard log directory
    logdir = r'.\logs\scalars\{}'.format(time.time())
    # CALLBACKS
    # TensorBoard
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Save the best va_loss file
    #checkpointer = keras.callbacks.ModelCheckpoint(filepath=chkdir, verbose=1, save_best_only=True)
    # Define optimizers
    Adam = keras.optimizers.Adam(learning_rate=
                                 lr, beta_1=0.9, beta_2=0.999)
    RMSprop = keras.optimizers.RMSprop(learning_rate=1e-2, rho=0.9)
    # Train
    model.compile(optimizer=Adam, loss='mean_squared_error')
    hist = model.fit(tr_data, tr_labels, epochs=epochs,
                     validation_data=(va_data, va_labels), callbacks=[tensorboard_callback],
                     batch_size=batch_size)
    # Post training
    model.save(save_dir)

    keras.backend.clear_session()
    # loss = hist.history['loss']
    print('\n*****\nTraining run time for data set length {} is : {} sec\n*****'.format(336 + flags.augment_size,
                                                                                        time.time() - start))


def load_model(model, load_type, dir):
    if load_type == 'load_weights':
        model.load_weights(dir)
    if load_type == 'load_model':
        model = keras.models.load_model(dir)

    return model


def test_model(model, test_data, test_labels, augment_size=0, plot=True):
    start = time.time()
    predictions = model.predict([test_data])
    MSE = np.square(np.subtract(test_labels.reshape(int(len(test_labels)), 1), predictions)).mean()
    wavelength = sp.arange(500, 820, 20)
    cur_time = datetime.now().strftime("%Y%m%d-%H%M")

    if not os.path.isdir('./kfold-predictions-vs-wavelength/{}'.format(cur_time)):
        os.makedirs('./kfold-predictions-vs-wavelength/{}'.format(cur_time))

    if not os.path.isdir('./kfold-predictions-vs-labels/{}'.format(cur_time)):
        os.makedirs('./kfold-predictions-vs-labels/{}'.format(cur_time))

    if not os.path.isdir('./kfold-MSEs'.format(cur_time)):
        os.makedirs('./kfold-MSEs'.format(cur_time))

    for i in [0, 16, 32]:
        file = open('./kfold-predictions-vs-wavelength/{}/analyte_{}.txt'.format(cur_time,i),'a+')
        for p,l,w in zip(predictions[i:i+16], test_labels[i:i+16], wavelength):
            file.write('{}\t{}\t{}\n'.format(w, l[0],p[0]))
        file.close()

    file = open('./kfold-predictions-vs-labels/{}/graph_data.txt'.format(cur_time), 'a+')
    for p,l in zip(predictions, test_labels):
        file.write('{}\t{}\n'.format(l[0], p[0]))
    file.close()

    file = open('./kfold-MSEs/MSEs.txt','a+')
    file.write('MSE:\t{}\n'.format(MSE))
    file.close()

    logger.info('Augment size: {}'.format(augment_size))
    # MSE = sp.square(sp.subtract(test_labels, predictions)).mean()
    if plot == True:
        plt.scatter(test_labels, predictions, label='Predictions', c='red', alpha=0.5)
        plt.plot(test_labels, test_labels, label='Actual', c='green', alpha=0.7)
        legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
        legend.get_frame().set_facecolor('C7')
        plt.ylabel('Prediction Log10(loss)')
        plt.xlabel('Actual Log10(loss)')
        plt.grid()
        plt.title('Length of TR dataset 336 + {}'.format(augment_size))
        plt.show()

        for i in [0, 16, 32]:
            spl = UnivariateSpline(wavelength, predictions[i:i + 16])
            wavelength_smoothed = sp.linspace(500, 820, 100)
            spl.set_smoothing_factor(0.0)
            plt.plot(wavelength_smoothed, spl(wavelength_smoothed), label='Predictions', c='black', lw=2)
            plt.scatter(wavelength, test_labels[i:i + 16], label='Actual', c='red', alpha=0.7, marker='o')
            legend = plt.legend(loc='upper left', shadow=True, fontsize='large')
            legend.get_frame().set_facecolor('C7')
            plt.ylabel('Confinment loss in Log10(db/cm)')
            plt.xlabel('Wavelegth in nm')
            plt.grid()
            plt.title('Length of TR dataset 336 + {}'.format(augment_size))
            plt.show()
    print('\n*****\nTest run time of the ANN model is: {} sec\n*****\n'.format(time.time() - start))
    return predictions


############################### Low level optimized WGAN #####################

class Wgan_optim(object):
    def __init__(self, sess, flags, x, y):
        self.sess = sess
        self.flags = flags
        self.batch_size = flags.wgan_batch_size
        self.noise_dim = flags.noise_dim
        self.num_critic_input = flags.num_critic_input
        self.n_critic = flags.n_critic
        self.grad_penalty_weight = flags.grad_penalty_weight
        self.epochs = flags.wgan_epochs
        self.num_examples_to_generate = flags.num_examples_to_generate
        self.dataset = self.shape_wgan_data(x, y)
        self._build_net()
        self._logger()
        self._tensorboard()

    def _build_net(self):
        self.Y = tf.placeholder(tf.float32, shape=[None, self.num_critic_input], name='real_data')
        self.z = tf.placeholder(tf.float32, shape=[None, self.noise_dim], name='latent_vector')

        self.generator = self.make_generator
        self.discriminator = self.make_disc

        self.g_samples = self.generator(self.z)
        d_logit_real = self.discriminator(self.Y)
        d_logit_fake = self.discriminator(self.g_samples)

        # critic loss
        self.wgan_d_loss = tf.reduce_mean(d_logit_fake) - tf.reduce_mean(d_logit_real)
        # generator loss
        self.g_loss = -tf.reduce_mean(d_logit_fake)

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d_')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g_')

        # gradient penalty
        self.gp_loss = self.gradient_penalty()
        self.d_loss = self.wgan_d_loss + self.grad_penalty_weight * self.gp_loss

        # Optimizers for generator and discriminator
        self.gen_optim = tf.train.AdamOptimizer(
            learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)
        self.dis_optim = tf.train.AdamOptimizer(
            learning_rate=2e-4, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)

        self.saver = tf.train.Saver()
        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
        self.checkpoint_dir = "./training-wgan-checkpoints/{}".format(cur_time)
        if self.flags.k_fold:
            self.checkpoint_dir = "./training-kf-wgan-checkpoints/{}".format(cur_time)
        if not os.path.isdir(self.checkpoint_dir) and len(self.dataset):
            os.makedirs(self.checkpoint_dir)

    def gradient_penalty(self):
        alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
        differences = self.g_samples - self.Y
        interpolates = self.Y + (alpha * differences)
        gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        return gradient_penalty

    def make_generator(self, data, name='g_'):
        with tf.variable_scope(name):
            x = tf.layers.Dense(self.batch_size * (2 ** 2), name='l1', _reuse=tf.AUTO_REUSE)(data)
            x = tf.layers.BatchNormalization(name='b1')(x)
            x = tf.nn.relu(x, name='relu_1')

            x1 = tf.layers.Dense(self.batch_size * (2 ** 2), name='l2', _reuse=tf.AUTO_REUSE)(x)
            x1 = tf.layers.BatchNormalization(name='b2')(x1)
            x1 = tf.nn.relu(x1, name='relu_2')

            x2 = tf.layers.Dense(self.batch_size * (2 ** 2), name='l3', _reuse=tf.AUTO_REUSE)(x1)
            x2 = tf.layers.BatchNormalization(name='b3')(x2)
            x2 = tf.nn.relu(x2, name='relu_3')

            x3 = tf.layers.Dense(self.batch_size * (2 ** 2), name='l4', _reuse=tf.AUTO_REUSE)(x2)
            x3 = tf.layers.BatchNormalization(name='b4')(x3)
            x3 = tf.nn.relu(x3, name='relu_4')

            x4 = tf.layers.Dense(self.batch_size * (2 ** 2), name='l5', _reuse=tf.AUTO_REUSE)(x3)
            x4 = tf.layers.BatchNormalization(name='b5')(x4)
            x4 = tf.nn.relu(x4, name='relu_5')

            out = tf.layers.Dense(self.noise_dim, name='output', _reuse=tf.AUTO_REUSE)(x4)

            return out

    def make_disc(self, data, name='d_'):
        with tf.variable_scope(name) as scope:
            x = tf.layers.Dense(self.batch_size * (2 ** 2), name='l1', _reuse=tf.AUTO_REUSE)(data)
            x = tf.nn.leaky_relu(x, name='leaky_relu_1')

            x1 = tf.layers.Dense(self.batch_size * (2 ** 2), name='l2', _reuse=tf.AUTO_REUSE)(x)
            x1 = tf.nn.leaky_relu(x1, name='leaky_relu2')

            x2 = tf.layers.Dense(self.batch_size * (2 ** 1), name='l3', _reuse=tf.AUTO_REUSE)(x1)
            x2 = tf.nn.leaky_relu(x2, name='leaky_relu3')

            x3 = tf.layers.Dense(self.batch_size * (2 ** 0), name='l4', _reuse=tf.AUTO_REUSE)(x2)
            x3 = tf.nn.leaky_relu(x3, name='leaky_relu4')

            x4 = tf.layers.Dense(self.batch_size * (2 ** 0), name='l5', _reuse=tf.AUTO_REUSE)(x3)
            x4 = tf.nn.leaky_relu(x4, name='leaky_relu5')

            out = tf.layers.Dense(1, name='output', _reuse=tf.AUTO_REUSE, activation='linear')(x4)

            return out

    def shape_wgan_data(self, tr_data, tr_labels):
        if len(tr_data) == 0.:
            return []
        # BUFFER_SIZE = int(len(tr_data))
        # Place the labels and inputs in one vector
        dataset = sp.concatenate((tr_data, tr_labels), axis=1)
        # dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(self.batch_size)
        return dataset

    @tf.function
    def train_wgan(self):
        for epoch in range(self.epochs):
            start = time.time()
            for i in range(0, len(self.dataset) // self.batch_size):
                wgan_d_loss, gp_loss, d_loss = None, None, None
                str_slice = i * self.batch_size
                batch_y = self.dataset[str_slice:str_slice + self.batch_size]
                # train critic
                for _ in range(self.n_critic):
                    dis_feed = {self.z: self.sample_z(num=self.batch_size), self.Y: batch_y}
                    dis_run = [self.dis_optim, self.wgan_d_loss, self.gp_loss, self.d_loss]
                    _, wgan_d_loss, gp_loss, d_loss = self.sess.run(dis_run, feed_dict=dis_feed)
                # train generator
                gen_feed = {self.z: self.sample_z(num=self.batch_size), self.Y: batch_y}
                gen_run = [self.gen_optim, self.g_loss, self.g_samples, self.summary_op]
                _, g_loss, g_samples, summary = self.sess.run(gen_run, feed_dict=gen_feed)
                if (epoch + 1)  == self.flags.save_step:
                    # data_handler.plot_wgan(epoch + 1,)
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'wgan'), global_step=epoch)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            # print('\tGen_samples: ',g_samples)
            self.train_writer.add_summary(summary, epoch)
            self.train_writer.flush()

    def sample_z(self, num=12):
        return np.random.uniform(-1., 1., size=[num, self.flags.noise_dim])

    def _logger(self):
        logger.info('\n[*] batch_size: {}'.format(self.batch_size))
        logger.info('\n[*] n_critic: {}'.format(self.n_critic))
        logger.info('\n[*] gradient penalty weight_: {}'.format(self.grad_penalty_weight))
        logger.info('\n[*] epochs: {}'.format(self.flags.wgan_epochs))
        logger.warning(
            '\n[*] Number of layers: is set to 5 in both networks. To modify it, you need to do it manually in networky.Wgan_optim')

    def _tensorboard(self):
        tf.summary.scalar('loss/wgan_d_loss', self.wgan_d_loss)
        # tf.summary.scalar('loss/gp_loss', self.gp_loss)
        tf.summary.scalar('loss/critic_loss', self.d_loss)
        tf.summary.scalar('loss/g_loss', self.g_loss)

        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./ganlogs/{}".format(time.time()),
                                                  graph_def=self.sess.graph_def)

    @staticmethod
    def filter_1(data):
        # Noise filter for the SPR set
        cd1 = sum(int(o >= 1.0) for o in data[3:7])
        cd2 = sum(int(t <= 0.0) for t in p)
        return [cd1, cd2]

    @staticmethod
    def filter_2(data):
        # Noise filter for PCF set
        cd1 = sum(int(o <= 0.0) for o in data[:6])
        #nve = sum(int(t <= 0.0) for t in p[1:6])
        return cd1

    def generate_data(self,load=True):
        logger.info('\n*****\n GENERATING DATA ... \n****\n')
        start = time.time()

        if load:
            self.load_wgan()

        _df = sp.zeros(self.noise_dim).reshape(1, self.noise_dim)
        for _ in tqdm(range(self.flags.gen_iterations)):
            predictions = self.sess.run(self.g_samples,
                                        feed_dict={self.z: self.sample_z(num=self.num_examples_to_generate)})
            for p in predictions:
                if(elf.flags.data_set = "SPR"):
                    cd1, cd2 = self.filter_1(p)
                    if not cd1 and not cd2:
                        _df = np.concatenate((_df, p.reshape(1, self.noise_dim)), axis=0)
                if(self.flags.data_set = "PCF"):
                    cd1 = self.filter_2(p)
                    if not cd1:
                        _df = np.concatenate((_df, p.reshape(1, self.noise_dim)), axis=0)

        _df = pd.DataFrame(_df, index=None)
        _df = _df.drop(0, axis=0)
        _df.to_csv('.\gen_data\gen_data2.txt', index=False)
        logger.info('\n*****\nGeneration run time is: {} sec\n*****'.format(time.time() - start))

    def load_wgan(self):
        logger.info('* Reading checkpoint...')
        checkpoint_dir = self.flags.load_wga_model_dir
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        print(checkpoint_dir)
        print(checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            ckpt_name = os.path.basename(checkpoint.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

            meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            logger.info('* Load SUCCESS!')
            return True
        else:
            logger.info('* Load Failed')


############################# Wgan with KERAS. For comparasion purposes ############################

class Wgan_unoptim(object):
    def __init__(self, flags):
        self.flags = flags
        self.batch_size = flags.wgan_batch_size
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
        model.add(layers.Dense(self.batch_size * (2 ** 2), input_shape=(self.noise_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        # HIDDEN layers
        for _ in range(num_layers - 1):
            model.add(layers.Dense(self.batch_size * (2 ** 2)))
            if batch_norm:
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
        model.add(layers.Dense(self.batch_size * (2 ** 2), input_shape=(self.num_critic_input,)))
        model.add(layers.LeakyReLU())
        # HIDDEN LAYERS
        i = 2
        for _ in range(num_layers - 1):
            model.add(layers.Dense(self.batch_size * (2 ** i)))
            model.add(layers.LeakyReLU())
            i -= 1
            if i < 0:
                i = 0
        # OUTPUT layer
        model.add(layers.Dense(1, activation='linear'))

        return model

    def gradient_penalty(self, model, x_real, x_fake):
        alpha = tf.compat.v1.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
        diff = x_fake - x_real
        interpolates = x_real + (alpha * diff)
        gradients = tf.gradients(model(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    @tf.function
    def train_critic(self, data, generator, critic, epoch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as cr_tape, self.critic_summary_writer.as_default():
            generated_data = generator(noise, training=True)
            data = tf.dtypes.cast(data, tf.float32)
            real_output = critic(data, training=True)
            fake_output = critic(generated_data, training=True)
            critic_loss = self.critic_loss(real_output, fake_output)
            gp = self.gradient_penalty(critic, data, generated_data)
            critic_loss += self.grad_penalty_weight * gp
            # self.train_loss_cr(critic_loss)
            # tf.summary.scalar('loss', self.train_loss_cr.result(), step=epoch)
        gradients_of_critic = cr_tape.gradient(critic_loss, critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, critic.trainable_variables))

    @tf.function
    def train_gen(self, generator, critic, epoch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, self.gen_summary_writer.as_default():
            generated_data = generator(noise, training=True)
            fake_output = critic(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output)
            # self.train_loss_gen(gen_loss)
            # tf.summary.scalar('loss', self.train_loss_gen.result(), step=epoch)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    def shape_wgan_data(self, tr_data, tr_labels):
        BUFFER_SIZE = int(len(tr_data))
        # Place the labels and inputs in one vector
        dataset = sp.concatenate((tr_data, tr_labels), axis=1)
        train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(BUFFER_SIZE).batch(self.batch_size)

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

    def train_wgan(self, tr_data, tr_labels, generator, critic):
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
            # self.generate_and_save_data(1,training=True, generator=generator, critic=critic)
            # Plot generated data vs wavelength each 200 epoch
            if (epoch + 1) == 500:
                # data_handler.plot_wgan(epoch + 1,)
                checkpoint.save(file_prefix=self.checkpoint_prefix)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        print('\n*****\nTraining run time for the wgan is: {} sec\n****\n'.format(time.time() - start))

    def generate_and_save_data(self, iterations, training, generator, critic):
        print('\n*****\n GENERATING DATA ... \n****\n')

        start = time.time()
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.critic_optimizer,
                                         generator=generator,
                                         discriminator=critic)
        if training == False:
            checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        _df = sp.zeros(7).reshape(1, 7)
        for _ in tqdm(range(iterations)):
            seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
            predictions = generator(seed, training=training)
            print(predictions)
            if (training == False):
                for p in predictions:
                    # Noise filter
                    ones = sum(int(o >= 1.0) for o in p[1:6])
                    #zeros = sum(int(z < 0.1) for z in p)
                    # sp.savetxt(r'gen_data_pcf.txt', p, delimiter=',')
                    if (1):
                        _df = np.concatenate((_df, tf.reshape(p, [1, 7])), axis=0)
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
