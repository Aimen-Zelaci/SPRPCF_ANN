import os
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import networks, data_handler, initializer
from networks import Wgan_optim

# tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(16)
FLAGS = initializer.init()
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
sess = tf.Session(config=run_config)


def train_wgan():
    tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname=FLAGS.data)
    wgan = Wgan_optim(sess, FLAGS, tr_data, tr_labels)
    sess.run(tf.global_variables_initializer())
    wgan.train_wgan()


def generate_data():
    # WILL SAVE TO gen_data\gen_data2.txt // Data already generated in gen_data\gen_data.txt#
    wgan = Wgan_optim(sess, FLAGS, [], [])
    sess.run(tf.global_variables_initializer())
    wgan.generate_data()


def train_ann_model():
    ann_model = networks.make_model(FLAGS)
    # Load the same data used to train the WGAN
    # SPR-based PCF
    tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname=FLAGS.data)
    # PCF data
    # tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_pcf_data(fname=FLAGS.pcf_data)
    # Augment data. To train with only the original real samples, set augment_size = 0.
    # For different augmentations, change the batch_size/ learning_rate accordingly.
    tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, FLAGS)
    # Train
    networks.train_model(model=ann_model,
                         tr_data=tr_data,
                         tr_labels=tr_labels,
                         va_data=va_data,
                         va_labels=va_labels,
                         flags=FLAGS
                         )


def test_model():
    ann_model = networks.make_model(FLAGS)
    _, __, _, ___, test_data, test_labels = data_handler.load_data(FLAGS.shuffled_data)
    # _, __, _, __, test_data, test_labels = data_handler.load_pcf_data(FLAGS.shuffled_pcf_data)

    # LOAD_TYPE = load the whole model or load the best checkpoint(load weights)
    loaded_model = networks.load_model(model=ann_model,
                                       load_type='load_model',
                                       dir=FLAGS.model_to_test)
    predictions = networks.test_model(model=loaded_model,
                                      test_data=test_data,
                                      test_labels=test_labels,
                                      augment_size=FLAGS.augment_size,
                                      plot=True)


def main(_):
    # 1. TRAIN WGAN
    if FLAGS.train_wgan:
        train_wgan()
    # 2. GENERATE DATA
    if FLAGS.generate:
        generate_data()
    # 3. TRAIN ANN MODEL/Augment data
    if FLAGS.train_ann:
        train_ann_model()
    # 4. TEST
    if FLAGS.test_ann:
        test_model()


if __name__ == '__main__':
    tf.app.run()
