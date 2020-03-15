import networks, data_handler
from networks import Wgan
import time

# Size of data augmentation
AUGMENT_SIZE = 1000

# Load Original data
# OUR SPR PHOTONIC SENSOR
tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname='.\data\data.xlsx')

# THEIR PCF DATA
#tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_pcf_data()

# Augment data after training the wgan
# To augment our data use fname = '.\gen_data\gen_data.txt'
# To augment their data use fname = '.\gen_data\gen_data_pcf.txt'
#tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, AUGMENT_SIZE, fname='.\gen_data\gen_data_pcf.txt')

# Save / Checkpoint directory
save_dir = r'.\trained-nets\model1.h5'
chkdir = r'.\trained-weights\weights1.hdf5'

if __name__ == '__main__':
    # PLEASE TRAIN the ANN and WGAN seperately !

    # TRAIN WGAN
    start = time.time()
    wgan = Wgan(BATCH_SIZE = 12,
                noise_dim = 7,
                num_critic_input = 7,
                n_critic = 5,
                grad_penalty_weight = 10,
                num_examples_to_generate = 8)
    generator = wgan.make_generator_model(num_layers=4)
    critic = wgan.make_critic_model(num_layers=4)
    wgan.train_wgan(tr_data,tr_labels,epochs = 2000, generator=generator, critic=critic)
    print('Training time for the wgan is: {} sec'.format(time.time() - start))

    # TRAIN ANN MODEL
    ann_model = networks.make_model(num_layers=5,
                                    num_inputs=6,
                                    num_outputs=1,
                                    num_neurons=50)

    for augment_size in [0,1000,2000,3000]:
        tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, augment_size, fname='.\gen_data\gen_data.txt')
        start = time.time()
        networks.train_model(model = ann_model,
                             epochs=2000,
                             tr_data=tr_data,
                             tr_labels=tr_labels,
                             va_data=va_data,
                             va_labels=va_labels,
                             save_dir=r'.\trained-nets\model{}.h5'.format(augment_size),
                             chkdir=r'.\trained-weights\weights.hdf5'.format(augment_size)
                             )
        print('Training time for data set length {} is : {} sec'.format(336+augment_size, time.time() - start))

    # TEST
    # LOAD_TYPE = load the whole model or load the best checkpoint(load weights)
    start = time.time()
    ann_model = networks.load_model(model=ann_model,
                                    load_type='load_weights',
                                    dir=r'.\trained-weights\weights1')
    networks.test_model(model=ann_model,
                        test_data=test_data,
                        test_labels=test_labels)
    print('Test time is: {}'.format(time.time() - start))
