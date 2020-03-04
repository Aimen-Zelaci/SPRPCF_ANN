import data_handler
import networks

# Size of data augmentation
AUGMENT_SIZE = 1000

# Load Original data
# OUR SPR PHOTONIC SENSOR
#tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_data(fname='data.xlsx')

# THEIR PCF DATA
tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_handler.load_pcf_data()

# Augment data
#tr_data, tr_labels = data_handler.augment_data(tr_data, tr_labels, AUGMENT_SIZE, fname='gen_data.txt')

# Save / Checkpoint directory
save_dir = r'.\trained-nets\model1.h5'
chkdir = r'.\trained-weights\weights1.hdf5'

if __name__ == '__main__':
    #print(tr_data[-1])
    #networks.generate_and_save_data(training=False)
    networks.train_wgan(tr_data,tr_labels, epochs=200)
    #networks.train_model(EPOCHS, tr_data, tr_labels, va_data, va_labels, save_dir, chkdir)
