import tensorflow.compat.v1 as tf
import networks, data_handler, initializer
from networks import Wgan
import time

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(16)
FLAGS = initializer.init()

wgan = Wgan(FLAGS)

generator = wgan.make_generator_model()
critic = wgan.make_critic_model()

ann_model = networks.make_model(FLAGS)
def train_wgan():
    start = time.time()
    tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname=FLAGS.data)
    wgan.train_wgan(tr_data,tr_labels, generator=generator, critic=critic)

def generate_data():
    # WILL SAVE TO gen_data\gen_data2.txt // Data already generated in gen_data\gen_data.txt#
    wgan.generate_and_save_data(iterations = FLAGS.gen_iterations, training=False, generator=generator, critic=critic)

def train_ann_model():
    # Load the same data used to train the WGAN
    tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname=FLAGS.shuffled_data)
    # Augment data. To train with only the original real samples, set augment_size = 0.
    # For different augmentations, change the batch_size/ learning_rate accordingly.
    tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, FLAGS)
    # Train
    networks.train_model(model = ann_model,
                         tr_data=tr_data,
                         tr_labels=tr_labels,
                         va_data=va_data,
                         va_labels=va_labels,
                         flags=FLAGS
                         )

def test_model():
    _, __, _, ___, test_data, test_labels = data_handler.load_data(FLAGS.shuffled_data)
    # LOAD_TYPE = load the whole model or load the best checkpoint(load weights)
    loaded_model = networks.load_model(model=ann_model,
                                    load_type='load_model',
                                    dir=FLAGS.model_to_test)
    predictions = networks.test_model(model=loaded_model,
                        test_data=test_data,
                        test_labels=test_labels,
                        plot = True)

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
