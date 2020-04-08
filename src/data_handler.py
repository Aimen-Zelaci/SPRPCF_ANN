import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import initializer

FLAGS = initializer.init()

def split_data(x, y, kf=False):
    if kf:
        x = x.reshape(8, 3, 16, 6)
        y = y.reshape(8, 3, 16, 1)
    else:
        x = x.reshape(9, 3, 16, 6)
        y = y.reshape(9, 3, 16, 1)

    if FLAGS.k_fold and not kf:
        return [x,y]

    tr_data = x[:7].reshape(7 * 16 * 3, 6)
    tr_labels = y[:7].reshape(7 * 16 * 3, 1)
    #Shuffle tr data
    merge = np.concatenate((tr_data, tr_labels), axis=1)
    sp.random.shuffle(merge)
    tr_data = merge[:, :6].reshape(7 * 16 * 3, 6)
    tr_labels = merge[:, -1].reshape(7 * 16 * 3, 1)

    if kf:
        va_data = x[-1].reshape(1 * 16 * 3, 6)
        va_labels = y[-1].reshape(1 * 16 * 3, 1)
        return [tr_data, tr_labels, va_data, va_labels]
    else:
        va_data = x[-2].reshape(1 * 16 * 3, 6)
        va_labels = y[-2].reshape(1 * 16 * 3, 1)
        test_data = x[-1].reshape(1 * 16 * 3, 6)
        test_labels = y[-1].reshape(1 * 16 * 3, 1)
        return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]


def load_data(fname):
    data = pd.read_excel(fname)
    df = data.values
    print(data.head())

    if fname == r'./data/shuffled_df.xlsx':
        df = df.reshape(432, 7)
        x = df[:, :6]
        y = df[:, -1]
        # The training data is already shuffled, set shuffle_tr=False
        return split_data(x, y, )

    df = df.reshape(9, 3, 16, 7)
    sp.random.shuffle(df)
    df = df.reshape(432, 7)
    #analytes = df[:, 0]
    #analytes = (analytes * 100) % 10
    x = df[:, :6]
    x = x.reshape(432, 6)
    #analytes = analytes.reshape(432, 1)
    #x = sp.concatenate((analytes, x), axis=1)
    #x /= 10
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    y = df[:, -1]
    y = y * (10 ** 8)
    y = sp.log10(y).reshape(432, 1)

    df = np.concatenate((x, y), axis=1)
    # Save shuffled data frame
    df = pd.DataFrame(df, columns=['Analytes', 'lambda', 'Pitch', 'd1', 'd2', 'd3', 'loss'])
    df.to_excel(r'./data/shuffled_df.xlsx', index=False)

    return split_data(x, y)


def load_pcf_data(fname='.\data\pcf_data1.xlsx'):
    df = pd.read_excel(fname)
    df = df.values

    if fname != r'.\data\shuffled_pcf_df.xlsx':
        sp.random.shuffle(df)

    tr_data = df[:1000, :6].reshape(1000, 6)
    tr_labels = df[:1000, -1].reshape(1000, 1)

    va_data = df[1000:1050, :6].reshape(50, 6)
    va_labels = df[1000:1050, -1].reshape(50, 1)

    test_data = df[1050:, :6]
    test_labels = df[1050:, -1]

    df = pd.DataFrame(df)
    df.to_excel(r'.\data\shuffled_pcf_df.xlsx', index=False)

    return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]


# Augment data
def augment_data(tr_data, tr_labels):
    size = FLAGS.augment_size
    fname = FLAGS.gen_data_dir

    generated_data = pd.read_csv(fname).values
    # start_slice = size-1000

    if (size == 0):
        return [tr_data, tr_labels]

    gen_x = generated_data[:size, :6].reshape(size, 6)
    gen_y = generated_data[:size, -1].reshape(size, 1)
    # Concatenate arrays
    tr_labels = np.concatenate((tr_labels, gen_y), axis=0)
    tr_data = np.concatenate((tr_data, gen_x), axis=0)

    return [tr_data, tr_labels]


# Plot wgan progress
def plot_wgan(epoch, fname):
    real_data_fname = r'./data/data.xlsx'
    x_real, y_real, _, __, ___, ____ = load_data(real_data_fname)

    data = pd.read_csv(fname).values

    if len(data) == 0:
        return 0

    x_gen = data[:, 1]
    y_gen = data[:, -1]

    plt.margins(x=0.3, y=0.5)
    plt.scatter(x_gen, y_gen, marker='v', label='Predictions', alpha=0.7)
    plt.scatter(x_real, y_real, marker='o', label='Actual', alpha=0.5)
    legend = plt.legend(loc='left center', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('C7')
    plt.title('Epoch {}'.format(epoch))
    plt.xlabel('Wavelength')
    plt.ylabel('log10(loss)')
    plt.grid()
    plt.show()
