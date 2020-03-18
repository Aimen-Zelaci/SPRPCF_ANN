from __future__ import print_function
import data_handler
# Import MNIST data
import tensorflow.compat.v1 as tf
import scipy as sp
import time
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

tf.disable_eager_execution()

# Print CPUs / GPUs available
print(device_lib.list_local_devices())


# Load data
tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname='.\data\data.xlsx')

# Augment

#tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, 1000, fname='.\gen_data\gen_data.txt')

# Parameters
learning_rate = 0.0001
training_epochs = 200
batch_size = 8

# Network Parameters
n_hidden = 50
n_input = 6
n_outputs = 1
num_hidden_layers = 6

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


def initializer():
    weights = {
        'h1': tf.Variable(weight_initializer([n_input, n_hidden])),
        'out': tf.Variable(weight_initializer([n_hidden, n_outputs]))
    }
    biases = {
        'b1': tf.Variable(bias_initializer([n_hidden])),
        'out': tf.Variable(bias_initializer([n_outputs]))
    }

    # Create hidden layers
    for i in range(1,num_hidden_layers):
        weights['h{}'.format(i+1)] = tf.Variable(weight_initializer([n_hidden, n_hidden]))
        biases['b{}'.format(i+1)] = tf.Variable(bias_initializer([n_outputs]))

    return [weights, biases]

weights, biases = initializer()

# Create model
def multilayer_perceptron(x, weights, biases):
    in_x = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    in_x = tf.nn.relu(in_x)
    in_x = tf.layers.batch_normalization(in_x)

    for i in range(1,num_hidden_layers):
        in_x = tf.add(tf.matmul(in_x, weights['h{}'.format(i+1)]), biases['b{}'.format(i+1)])
        in_x = tf.nn.relu(in_x)
        in_x = tf.layers.batch_normalization(in_x)

    out_layer = tf.matmul(in_x, weights['out']) + biases['out']
    out_layer = tf.nn.relu(out_layer)

    return out_layer

# Initializing the variables
X = tf.placeholder("float", [None, n_input], name='X')
Y = tf.placeholder("float", [None, n_outputs], name='Y')

out = multilayer_perceptron(X, weights, biases)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(Y, out)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

start = time.time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

@tf.function
def train():
# Training cycle
    print('\n\n **** Training on {} samples'.format(int(len(tr_data))))
    for epoch in range(training_epochs):
        start_epoch = time.time()
        avg = 0.

        shuffle_indices = sp.random.permutation(sp.arange(len(tr_labels)))
        X_train = tr_data[shuffle_indices]
        y_train = tr_labels[shuffle_indices]
        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            cost = sess.run(loss_op, feed_dict={X:batch_x, Y:batch_y})
            avg+=cost

        va_mse = sess.run(loss_op, feed_dict={X:va_data, Y:va_labels})
        print('tr_loss at EPOCH {} is {} \t\t va_MSE is {} \t\t {} sec'.format(epoch+1, avg / (len(y_train) // batch_size), va_mse, time.time() - start_epoch ))

train()

print("Training Finished! {} sec".format( time.time()- start))
