"""
    ANN system for pcf data prediction.
    Copyright (C) 2021  Aimen Zelaci

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
import initializer

FLAGS = initializer.init()
num_inputs  = FLAGS.num_inputs
no_spr_samples = FLAGS.no_spr_samples
no_spr_analytes = FLAGS.no_spr_analytes
no_spr_configs = FLAGS.no_spr_configs

def split_data(x, y, kf=False):
    if kf:
        x = x.reshape(no_spr_configs -  1, no_spr_analytes, no_spr_samples, num_inputs)
        y = y.reshape(no_spr_configs - 1, no_spr_analytes, no_spr_samples, 1)
    else:
        x = x.reshape(no_spr_configs, no_spr_analytes, no_spr_samples, num_inputs)
        y = y.reshape(no_spr_configs, no_spr_analytes, no_spr_samples, 1)

    if FLAGS.k_fold and not kf:
        return [x,y]

    tr_data = x[: no_spr_configs -  2].reshape((no_spr_configs -  2) * no_spr_samples * no_spr_analytes, num_inputs)
    tr_labels = y[: no_spr_configs -  2].reshape((no_spr_configs -  2) * no_spr_samples * no_spr_analytes, 1)

    if kf:
        va_data = x[-1].reshape(1 * no_spr_samples * no_spr_analytes, num_inputs)
        va_labels = y[-1].reshape(1 * no_spr_samples * no_spr_analytes, 1)
        return [tr_data, tr_labels, va_data, va_labels]
    else:
        va_data = x[-2].reshape(1 * no_spr_samples * no_spr_analytes, num_inputs)
        va_labels = y[-2].reshape(1 * no_spr_samples * no_spr_analytes, 1)
        test_data = x[-1].reshape(1 * no_spr_samples * no_spr_analytes, num_inputs)
        test_labels = y[-1].reshape(1 * no_spr_samples * no_spr_analytes, 1)
        return [tr_data, tr_labels, va_data, va_labels, test_data, test_labels]


def load_spr_data(fname):
    data = pd.read_excel(fname)
    df = data.values
    print(data.head())

    if fname == r'./data/shuffled_df.xlsx':
        df = df.reshape(no_spr_configs * no_spr_samples * no_spr_analytes, no_spr_configs)
        x = df[:, :num_inputs]
        y = df[:, -1]
        # The training data is already shuffled, set shuffle_tr=False
        return split_data(x, y )

    df = df.reshape(no_spr_configs, no_spr_analytes, no_spr_samples,  -1)
    sp.random.shuffle(df)
    df = df.reshape(no_spr_configs * no_spr_samples * no_spr_analytes, -1)
    '''
    Tried different normalization techniques
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    print(x)
    x = normalize(x, norm='l2')
    pt = PowerTransformer()
    x = pt.fit_transform(x)
    scaler = StandardScaler(with_mean=False)
    df_temp = scaler.fit_transform(df)
    df[:,0] = df_temp[:,0]
    '''
    x = df[:, :num_inputs]
    x = x.reshape(432, num_inputs)

    y = df[:, -2]
    y = y * (10 ** 8)
    y = sp.log10(y).reshape(432, 1)

    df = np.concatenate((x, y), axis=1)
    # Save the shuffled data frame to map back the results after the experiments are over: for analysis purposes
    # df = pd.DataFrame(df, columns=['Analytes', 'Re(neff)','lambda', 'Pitch', 'd1', 'd2', 'd3', 'im(neff)'])
    # df.to_excel(r'./data/shuffled_spr_df.xlsx', index=False)

    return split_data(x, y)


def load_pcf_data(fname='.\data\pcf_data1.xlsx'):
    df = pd.read_excel(fname)
    df = df.values
    sp.random.shuffle(df)
    x = df[:,:6]
    y = df[:,-1]

    if FLAGS.k_fold_pcf2:
        return [x,y]

    if fname != r'.\data\shuffled_pcf_df.xlsx':
        sp.random.shuffle(df)

    # Save the shuffled data frame to map back the results after the experiments are over: for analysis purposes
    # df = pd.DataFrame(df)
    # df.to_excel(r'.\data\shuffled_pcf_df.xlsx', index=False)


# Augment data
def augment_data(tr_data, tr_labels):
    size = FLAGS.augment_size
    fname = FLAGS.gen_data_dir

    generated_data = pd.read_csv(fname).values
    # start_slice = size-1000

    if (size == 0):
        return [tr_data, tr_labels]

    gen_x = generated_data[:size, :num_inputs].reshape(size, num_inputs)
    gen_y = generated_data[:size, -1].reshape(size, 1)
    # Concatenate arrays
    tr_labels = np.concatenate((tr_labels, gen_y), axis=0)
    tr_data = np.concatenate((tr_data, gen_x), axis=0)

    return [tr_data, tr_labels]
