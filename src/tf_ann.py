from __future__ import print_function
import time
import data_handler
import initializer
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.python.client import device_lib

tf.disable_eager_execution()

FLAGS = initializer.init()


class Ann():
    def __init__(self, session, flags, tr_data, tr_labels, va_data, va_labels, test_data):
        self.learning_rate = flags.lr
        self.training_epochs = flags.epochs
        self.batch_size = flags.ANN_batch_size
        self.n_hidden = flags.num_neurons
        self.n_input = flags.num_inputs
        self.n_outputs = flags.num_outputs
        num_hidden_layers = flags.num_layers
        self.sess = session
        self._build_net()
        self.tr_data , self.tr_labels, self.va_data, self.va_labels, self.test_data = \
        tr_data , tr_labels, va_data, va_labels, test_data

    def _build_net(self):
        self.X = tf.placeholder("float", [None, self.n_input], name='X')
        self.Y = tf.placeholder("float", [None, self.n_outputs], name='Y')
        self.model = self.make_model
        self.out = self.model(self.X)
        # Define loss and optimizer
        self.loss_op = tf.losses.mean_squared_error(self.Y, self.out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)


    def make_model(self,data, name='m_'):
        with tf.variable_scope(name):
            x = tf.layers.Dense(self.n_hidden, name='l1', _reuse=tf.AUTO_REUSE)(data)
            x = tf.layers.BatchNormalization(name='b1')(x)
            x = tf.nn.relu(x, name='relu_1')

            x1 = tf.layers.Dense(self.n_hidden, name='l2', _reuse=tf.AUTO_REUSE)(x)
            x1 = tf.layers.BatchNormalization(name='b2')(x1)
            x1 = tf.nn.relu(x1, name='relu_2')

            x2 = tf.layers.Dense(self.n_hidden, name='l3', _reuse=tf.AUTO_REUSE)(x1)
            x2 = tf.layers.BatchNormalization(name='b3')(x2)
            x2 = tf.nn.relu(x2, name='relu_3')

            x3 = tf.layers.Dense(self.n_hidden, name='l4', _reuse=tf.AUTO_REUSE)(x2)
            x3 = tf.layers.BatchNormalization(name='b4')(x3)
            x3 = tf.nn.relu(x3, name='relu_4')

            x4 = tf.layers.Dense(self.n_hidden, name='l5', _reuse=tf.AUTO_REUSE)(x3)
            x4 = tf.layers.BatchNormalization(name='b5')(x4)
            x4 = tf.nn.relu(x4, name='relu_5')

            out = tf.layers.Dense(self.n_outputs, name='output', _reuse=tf.AUTO_REUSE)(x4)

            return tf.nn.relu(out, name='relu_out')

    @tf.function
    def train(self):
        # Training cycle

        print('\n\n **** Training on {} samples'.format(int(len(self.tr_data))))
        for epoch in range(self.training_epochs):
            start_epoch = time.time()
            avg = 0.
            # Minibatch training
            for i in range(0, len(self.tr_data) // self.batch_size):
                start = i * self.batch_size
                batch_x = self.tr_data[start:start + self.batch_size]
                batch_y = self.tr_labels[start:start + self.batch_size]

                # Run optimizer with batch
                _, cost = self.sess.run([self.train_op, self.loss_op], feed_dict={self.X: batch_x, self.Y: batch_y})
                avg += cost

            avg /= (len(self.tr_data) // self.batch_size)
            va_mse = self.sess.run(self.loss_op, feed_dict={self.X: self.va_data, self.Y: self.va_labels})

            print('tr_loss at EPOCH {} is {:.6f} \t\t va_MSE is {:.6f} \t\t {:.6f} sec'.format(epoch + 1, avg, va_mse,
                                                                                               time.time() - start_epoch))

        pred = self.sess.run(self.out, feed_dict={self.X: self.test_data})
        return pred

    print("Training Finished")
