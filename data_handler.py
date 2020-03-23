import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

def load_pcf_data(fname='pcf_data1.xlsx'):
    df_1 = pd.read_excel(fname)
    datafile_1 = df_1.values

    tr_data = datafile_1[:1000,2:6].reshape(1000,4)
    tr_labels = datafile_1[:1000,-1].reshape(1000,1)

    va_data = datafile_1[1000:1050,2:6].reshape(50,4)
    va_labels = datafile_1[1000:1050, -1].reshape(50, 1)

    test_data =datafile_1[1050:,2:6]
    test_labels =datafile_1[1050:,-1]

    return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]

def load_data(fname='data.xlsx'):
    data =pd.read_excel(fname)
    df = data.values

    analytes = df[:,0]
    analytes = (analytes*100)%10
    x = df[:, 1:6]
    
    x = x.reshape(432,5)
    analytes = analytes.reshape(432,1)
    
    x = sp.concatenate((analytes, x), axis=1)
    x /= 10

    y = df[:, -1]
    y = y * (10**8)
    y = sp.log10(y)
    #Assert shape
    x = x.reshape(432,6)
    y = y.reshape(432,1)

    tr_data = x[:336]
    tr_labels = y[:336]

    va_data = x[336:384]
    va_labels = y[336:384]

    test_data = x[384:]
    test_labels = y[384:]

    return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]

# Augment data
def augment_data(tr_data, tr_labels, size=1000, fname='gen_data.txt'):
    generated_data = pd.read_csv(fname).values

    #OUR data
    if(fname=='gen_data'):
        gen_x = generated_data[:size, :6]
       
    #THEIR data
    if(fname=='gen_data_pcf'):
        gen_x = generated_data[:size, :4]
        
    gen_y = generated_data[:size, -1]
    
    # Concatenate arrays
    tr_labels = sp.concatenate((tr_labels, gen_y), axis=0)
    tr_data = sp.concatenate((tr_data, gen_x), axis=0)

    # Assert shape
    tr_data = sp.array([x.reshape(6, ) for x in tr_data]).reshape(int(len(tr_data)), 6)
    tr_labels = sp.array([y.reshape(1, ) for y in tr_labels]).reshape(int(len(tr_data)))

    return [tr_data, tr_labels]

# Plot wgan progress
def plot_wgan(epoch, fname):
    real_data_fname = r'data.xlsx'
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