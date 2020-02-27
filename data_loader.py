import scipy as sp
import random
import matplotlib.pyplot as plt
import os

NUM_ANALYTES = 3
NUM_DATA_POINTS = 16

configs = [
    # Training configs
    [1.5, 2.5, 7.5, 3.5],
    [2.4, 2.5, 5.5, 1.5],
    [2.0, 4.5, 7.5, 3.5],
    [1.5, 4.5, 7.5, 3.5],
    [1.5, 2.5, 5.5, 1.5],
    [2.0, 2.5, 7.5, 3.5],
    [2.4, 2.5, 7.5, 3.5],
    # Validation config
    [2.4, 4.5, 7.5, 3.5],
    # Test config
    [2.0, 2.5, 5.5, 1.5],
]

lmbdas = []
losses = []

for j in [5, 3, 7, 4, 6, 8, 2, 1, 9]:
    for i in [3, 4, 5]:
        BASE_FOLDER = r'\data\config{}'.format(str(j))
        BASE_NAME = r'13{}.tsv'.format(str(i))
        fname = os.path.join(BASE_FOLDER, BASE_NAME)
        # For relatively small data size
        data = sp.genfromtxt(fname, delimiter='\t')
        x = data[:, 0]
        y = data[:, 1]
        y = y * (10 ** 7)
        y = sp.log10(y)
    lmbdas.append(x)
    losses.append(y)


def shape_data(configs, analytes, losses, lmbdas):
    analytes = [100 * analyte for analyte in analytes]
    x_shaped = sp.array(sp.zeros(NUM_ANALYTES * 6 * 16)).reshape(NUM_ANALYTES * 16, 6)
    y = sp.array(sp.zeros(NUM_ANALYTES * NUM_DATA_POINTS)).reshape(NUM_ANALYTES * NUM_DATA_POINTS, 1)
    k = NUM_DATA_POINTS
    j = 0
    aIndex = 0
    while (k <= NUM_ANALYTES * NUM_DATA_POINTS):
        while (j < k):
            temp = sp.array(sp.zeros(6))
            firstIndex = int(analytes[aIndex]) % 10
            temp[0] = firstIndex
            temp[1] = lmbdas[j]
            for i in range(4):
                temp[i + 2] = configs[i]
            temp.reshape(1, 6)
            x_shaped[j] = temp
            loss = losses[j]
            y[j] = loss
            j = j + 1
        k += NUM_DATA_POINTS
        aIndex += 1
    x_shaped = x_shaped / 10
    data = [(x, y) for x, y in zip(x_shaped, y)]

    return data

def load_data(model):
    # Number of training configurations
    NUM_CONFIGS = 7

    analytes = [1.33, 1.34, 1.35]
    dataset = []

    for param, lmbda, loss in zip(configs, lmbdas, losses):
        dataset += shape_data(param, analytes, loss, lmbda)
    dataset = [(x.reshape(6, 1), y.reshape(1, 1)) for x, y in dataset]
    trn_data = dataset[:NUM_DATA_POINTS * NUM_ANALYTES * NUM_CONFIGS]

    # Shuffle tr data
    random.shuffle(trn_data)

    train_data = []
    train_labels = []
    va_data = []
    va_labels = []
    test_data = []
    test_labels = []

    # Shape data for Keras requirements
    if model == 'keras':
        # Training data
        for x, y in trn_data:
            train_data.append(x)
            train_labels.append(y)
        train_data = sp.array([x.flatten() for x in train_data])
        train_data = train_data.reshape(NUM_DATA_POINTS * NUM_ANALYTES * NUM_CONFIGS, 6)
        train_labels = sp.array([y.flatten() for y in train_labels])

        # Validation data
        for x, y in dataset[NUM_DATA_POINTS * NUM_ANALYTES * 7:NUM_DATA_POINTS * NUM_ANALYTES * 8]:
            va_data.append(x)
            va_labels.append(y)
        validation_data = sp.array([x.flatten() for x in va_data])
        validation_data = validation_data.reshape(NUM_DATA_POINTS * NUM_ANALYTES, 6)
        validation_labels = sp.array([y.flatten() for y in va_labels])

        # Test data
        for x, y in dataset[NUM_DATA_POINTS * NUM_ANALYTES * 8:]:
            test_data.append(x)
            test_labels.append(y)
        test_data = sp.array([x.flatten() for x in test_data])
        test_data = test_data.reshape(NUM_DATA_POINTS * NUM_ANALYTES, 6)
        test_labels = sp.array([y.flatten() for y in test_labels])

        return [train_data, train_labels, validation_data, validation_labels, test_data, test_labels]

    else:
        return dataset

# Augment data
def augment_data(tr_data, tr_labels, size, fname):
    generated_data = sp.genfromtxt(fname)
    random.shuffle(generated_data)

    tr_labels = [y for y in tr_labels]
    tr_data = [x for x in tr_data]

    # Generates samples
    _ = [x for x in generated_data[:size, :6]]

    # Generated labels
    __ = [y for y in generated_data[:size, -1]]

    # Join arrays
    tr_labels += __
    tr_data += _

    # Convert to numpy arrays
    tr_data = sp.array([x.reshape(6, ) for x in tr_data]).reshape(int(len(tr_data)), 6)
    tr_labels = sp.array([y.reshape(1, ) for y in tr_labels]).reshape(int(len(tr_data)))

    return [tr_data, tr_labels]

# Plot wgan progress
def plot_wgan(epoch, fname):
    # For relatively small data size
    data = sp.genfromtxt(fname)
    if len(data) == 0:
        return 0

    y = data[:, -1]
    x = data[:, 1]

    plt.margins(x=0.3, y=0.5)
    plt.scatter(x, y, marker='v', label='Predictions', alpha=0.7)
    plt.scatter(lmbdas[-1], losses[-1], marker='o', label='Actual', alpha=0.5)
    legend = plt.legend(loc='left center', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('C7')
    plt.title('Epoch {}'.format(epoch))
    plt.xlabel('Wavelength')
    plt.ylabel('log10(loss)')
    plt.grid()
    plt.show()
