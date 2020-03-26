import networks, data_handler
from networks import Wgan
import time

wgan = Wgan(BATCH_SIZE = 12,
            noise_dim = 7,
            num_critic_input = 7,
            n_critic = 5,
            grad_penalty_weight = 10,
            num_examples_to_generate = 8)

generator = wgan.make_generator_model(num_layers=5, batch_norm=True)
critic = wgan.make_critic_model(num_layers=5)

ann_model = networks.make_model(num_layers=6,
                                num_inputs=6,
                                num_outputs=1,
                                num_neurons=50,
                                batch_norm=True)
def train_wgan():
    start = time.time()

    tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname='.\data\data.xlsx')

    print('\n*****\nTraining The WGAN  \n****\n')
    wgan.train_wgan(tr_data,tr_labels,epochs = 2000, generator=generator, critic=critic)
    print('\n*****\nTraining run time for the wgan is: {} sec\n****\n'.format(time.time() - start))

def generate_data():
    print('\n*****\n GENERATING DATA ... \n****\n')
    start = time.time()
    # WILL SAVE TO gen_data\gen_data2.txt // Data already generated in gen_data\gen_data.txt#
    wgan.generate_and_save_data(iterations = 1000, training=False, generator=generator, critic=critic)
    print('\n*****\nGeneration run time is: {} sec\n*****'.format(time.time() - start))

def train_ann_model():
    start = time.time()
    # Load the same data used to train the WGAN
    tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname='.\data\shuffled_df.xlsx')

    # Augment data. To train with only the original real samples, set augment_size = 0.
    # For different augmentations, change the batch_size/ learning_rate accordingly.
    augment_size = 1000
    tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, augment_size, fname='.\gen_data\gen_data.txt')
    # Train
    print('\n\n Training the ANN model on {} samples \n\n'.format(int(len(tr_data))))
    networks.train_model(model = ann_model,
                         epochs=2000,
                         batch_size = 8,
                         learning_rate = 1e-4,
                         tr_data=tr_data,
                         tr_labels=tr_labels,
                         va_data=va_data,
                         va_labels=va_labels,
                         save_dir=r'.\trained-nets\model{}.h5'.format(augment_size),
                         chkdir=r'.\trained-weights\weights{}.hdf5'.format(augment_size)
                         )

    print('\n*****\nTraining run time for data set length {} is : {} sec\n*****'.format(336+augment_size, time.time() - start))

def test_model():
    start = time.time()
    _, __, _, ___, test_data, test_labels = data_handler.load_data(fname='.\data\shuffled_df.xlsx')
    # LOAD_TYPE = load the whole model or load the best checkpoint(load weights)
    loaded_model = networks.load_model(model=ann_model,
                                    load_type='load_model',
                                    dir=r'.\trained-weights\weights0.hdf5')
    predictions = networks.test_model(model=loaded_model,
                        test_data=test_data,
                        test_labels=test_labels,
                        plot = True)
    print('\n*****\nTest run time of the ANN model is: {} sec\n*****\n'.format(time.time() - start))

if __name__ == '__main__':
    # PLEASE TRAIN the ANN and WGAN seperately !
    # 1. TRAIN WGAN
    train_wgan()

    # 2. GENERATE DATA
    generate_data()

    # 3. TRAIN ANN MODEL/Augment data
    train_ann_model()

    # 4. TEST
    test_model()

