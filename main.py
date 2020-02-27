import data_loader
import networks

# Size of data augmentation
AUGMENT_SIZE = 1000

EPOCHS = 2000

# Load Original data
tr_data, tr_labels, va_data, va_labels, test_data, test_labels = data_loader.load_data('keras')

# Augment data
fname = r'\gen_data.txt'
tr_data, tr_labels = data_loader.augment_data(tr_data, tr_labels, AUGMENT_SIZE, fname)

# Save / Checkpoint directory
save_dir = r'\trained-nets\last.h5'
chkdir = r'\trained-weights\last.hdf5'

if __name__ == '__main__':
    networks.train_model(EPOCHS, tr_data, tr_labels, va_data, va_labels, save_dir, chkdir)
