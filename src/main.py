import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import networks, data_handler, initializer
from networks import Wgan_optim
from sklearn.model_selection import KFold

# tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(16)
FLAGS = initializer.init()
#tr_data, tr_labels, va_data, va_labels, test_data, test_labels = [],[],[],[],[],[]

def train_wgan(tr_data=[], tr_labels=[],generate=False):
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    if not FLAGS.k_fold:
        tr_data, tr_labels, __, _, __, _ = data_handler.load_data(fname=FLAGS.data)
    #tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_pcf_data(fname=FLAGS.pcf_data)
    wgan = Wgan_optim(sess, FLAGS, tr_data, tr_labels)
    sess.run(tf.global_variables_initializer())
    wgan.train_wgan()
    if generate:
        wgan.generate_data(load=False)
    tf.reset_default_graph()

def generate_data(sess):
    # WILL SAVE TO gen_data\gen_data2.txt // Data already generated in gen_data\gen_data.txt#
    wgan = Wgan_optim(sess, FLAGS, [], [])
    sess.run(tf.global_variables_initializer())
    wgan.generate_data()


def train_ann_model(tr_data=[], tr_labels=[], va_data=[], va_labels=[]):
    ann_model = networks.make_model(FLAGS)
    # Load the same data used to train the WGAN
    # SPR-based PCF
    if not FLAGS.k_fold:
        tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname=FLAGS.data)
    # PCF data
    # tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_pcf_data(fname=FLAGS.shuffled_pcf_data)
    # Augment data. To train with only the original real samples, set augment_size = 0.
    # For different augmentations, change the batch_size/ learning_rate accordingly.
    tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels)
    # Train
    networks.train_model(model=ann_model,
                         tr_data=tr_data,
                         tr_labels=tr_labels,
                         va_data=va_data,
                         va_labels=va_labels,
                         flags=FLAGS
                         )


def test_model(test_data=[], test_labels=[], plot=False):
    ann_model = networks.make_model(FLAGS)
    if not FLAGS.k_fold:
        _, __, _, ___, test_data, test_labels = data_handler.load_data(FLAGS.shuffled_data)
    # _, __, _, __, test_data, test_labels = data_handler.load_pcf_data(FLAGS.shuffled_pcf_data)

    # LOAD_TYPE = load the whole model or load the best checkpoint(load weights)
    loaded_model = networks.load_model(model=ann_model,
                                       load_type='load_weights',
                                       dir=FLAGS.model_to_test)
    predictions = networks.test_model(model=loaded_model,
                                      test_data=test_data,
                                      test_labels=test_labels,
                                      augment_size=FLAGS.augment_size,
                                      plot=plot)


def train_system_kf():
    kf = KFold(9)
    x, y = data_handler.load_data(fname=FLAGS.data)

    for tr_index, test_index in kf.split(x):
        X_train, X_test = x[tr_index], x[test_index]
        y_train, y_test = y[tr_index], y[test_index]
        tr_data, tr_labels, va_data, va_labels = data_handler.split_data(X_train, y_train, kf=True)
        test_data, test_labels = X_test.reshape(16*3,6) , y_test.reshape(16*3,1)

        print(tr_data.shape)
        print(va_data.shape)
        print(test_data.shape)
        print('\ntrain/va - folds:\t', X_train.shape[0])
        print('\ntest - folds:\t', X_test.shape[0])
        train_wgan(tr_data, tr_labels, generate=True)
        train_ann_model(tr_data, tr_labels, va_data, va_labels)
        test_model(test_data, test_labels, False)

def main(_):
    # 1. TRAIN WGAN
    if FLAGS.train_wgan:
        train_wgan()
    # 2. GENERATE DATA
    if FLAGS.generate:
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        sess = tf.Session(config=run_config)
        generate_data(sess)
    # 3. TRAIN ANN MODEL/Augment data
    if FLAGS.train_ann:
        train_ann_model()
    # 4. TEST
    if FLAGS.test_ann:
        test_model()

    # KFOLD training
    if FLAGS.k_fold:
        train_system_kf()

if __name__ == '__main__':
    tf.app.run()
