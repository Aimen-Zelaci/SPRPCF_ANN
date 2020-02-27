import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def load_data(fname):
    data =pd.read_excel(fname)
    df = data.values
    x = df[:, :6]
    x = x / 10
    y = df[:, -1]
    y = sp.log10(y)

    tr_data = x[:337]
    tr_labels = y[:337]

    va_data = x[336:385]
    va_labels = y[336:385]

    test_data = x[385:]
    test_labels = y[385:]

    return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]

# Augment data
def augment_data(tr_data, tr_labels, size, fname):
    generated_data = pd.read_csv(fname).values

    gen_x = generated_data[:size, :6]
    gen_y = generated_data[:size, -1]

    # Concatenate arrays
    tr_labels = sp.concatenate(tr_labels, gen_y, axis=0)
    tr_data = sp.concatenate(tr_data, gen_x, axis=0)

    # Assert shape
    tr_data = sp.array([x.reshape(6, ) for x in tr_data]).reshape(int(len(tr_data)), 6)
    tr_labels = sp.array([y.reshape(1, ) for y in tr_labels]).reshape(int(len(tr_data)))

    return [tr_data, tr_labels]

# Plot wgan progress
def plot_wgan(epoch, fname):

    real_data_fname = r'data.xlsx'
    x_real, y_real ,_,__,_,__ = load_data(real_data_fname)

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
