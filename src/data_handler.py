import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_pcf_data(fname='.\data\pcf_data1.xlsx'):
    df_1 = pd.read_excel(fname)
    datafile_1 = df_1.values

    sp.random.shuffle(datafile_1)

    tr_data = datafile_1[:1000, 2:6].reshape(1000,4)
    tr_labels = datafile_1[:1000,-1].reshape(1000,1)

    va_data = datafile_1[1000:1050, 2:6].reshape(50,4)
    va_labels = datafile_1[1000:1050, -1].reshape(50, 1)

    test_data =datafile_1[1050:, 2:6]
    test_labels =datafile_1[1050:,-1]

    return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]

def handle_data(x, y, shuffle_tr=True):
    x = x.reshape(9,3,16,6)
    y = y.reshape(9,3,16,1)

    tr_data = x[:7].reshape(7 * 16 * 3, 6)
    tr_labels = y[:7].reshape(7 * 16 * 3, 1)

    if shuffle_tr == True:
        merge = np.concatenate((tr_data,tr_labels), axis=1)
        sp.random.shuffle(merge)
        tr_data = merge[:,:6].reshape(7 * 16 * 3, 6)
        tr_labels = merge[:,-1].reshape(7 * 16 * 3, 1)

    va_data = x[-2].reshape(1*16*3,6)
    va_labels = y[-2].reshape(1*16*3,1)

    test_data = x[-1].reshape(1*16*3,6)
    test_labels = y[-1].reshape(1*16*3,1)

    return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]


def load_data(fname):
    data =pd.read_excel(fname)
    df = data.values
    print(data.head())

    if fname == r'.\data\shuffled_df.xlsx':
        df = df.reshape(432,7)
        x = df[:,:6]
        y = df[:,-1]
        # The training data is already shuffled, set shuffle_tr=False
        return handle_data(x,y, shuffle_tr=False)

    df = df.reshape(9,3,16,7)
    sp.random.shuffle(df)
    df = df.reshape(432,7)

    analytes = df[:,0]
    analytes = (analytes*100)%10
    x = df[:, 1:6]

    x = x.reshape(432,5)
    analytes = analytes.reshape(432,1)

    x = sp.concatenate((analytes, x), axis=1)
    x /= 10

    y = df[:, -1]
    y = y * (10**8)
    y = sp.log10(y).reshape(432,1)
    #Assert shape

    df = np.concatenate((x,y), axis=1)
    # Save shuffled data frame
    df = pd.DataFrame(df, columns=['Analytes','lambda','Pitch','d1','d2','d3','loss'])
    df.to_excel(r'.\data\shuffled_df.xlsx', index = False)

    return handle_data(x,y, shuffle_tr=True)


# Augment data
def augment_data(tr_data, tr_labels, flags):
    size = flags.augment_size
    fname = flags.gen_data_dir

    generated_data = pd.read_csv(fname).values
    #start_slice = size-1000

    if(size == 0):
        return [tr_data, tr_labels]

    #OUR data
    gen_x = generated_data[:size, :6].reshape(size, 6)
    #THEIR data
    if(fname=='gen_data_pcf'):
        gen_x = generated_data[:size, :4].reshape(size, 4)

    gen_y = generated_data[:size, -1].reshape(size, 1)

    # Concatenate arrays
    tr_labels = np.concatenate((tr_labels, gen_y), axis=0)
    tr_data = np.concatenate((tr_data, gen_x), axis=0)

    return [tr_data, tr_labels]

# Plot wgan progress
def plot_wgan(epoch, fname):
    real_data_fname = r'.\data\data.xlsx'
    x_real, y_real ,_,__,___,____ = load_data(real_data_fname)

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
