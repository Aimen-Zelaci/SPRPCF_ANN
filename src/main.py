from src import networks, data_handler
from src.networks import Wgan

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
    # TRAIN WGAN
    wgan = Wgan(BATCH_SIZE = 12,
                noise_dim = 7,
                num_critic_input = 7,
                n_critic = 5,
                grad_penalty_weight = 10,
                num_examples_to_generate = 8)
    generator = wgan.make_generator_model(num_layers=4)
    critic = wgan.make_critic_model(num_layers=4)
    #wgan.train_wgan(tr_data,tr_labels,epochs = 200, generator=generator, critic=critic)

    # TRAIN ANN MODEL
    ann_model = networks.make_model(num_layers=5,
                                    num_inputs=6,
                                    num_outputs=1,
                                    num_neurons=50)

    networks.train_model(model = ann_model,
                         epochs=2000,
                         tr_data=tr_data,
                         tr_labels=tr_labels,
                         va_data=va_data,
                         va_labels=va_labels,
                         save_dir=save_dir,
                         chkdir=chkdir)
    # TEST
    networks.test_model(model=ann_model,
                        test_data=test_data,
                        test_labels=test_labels)