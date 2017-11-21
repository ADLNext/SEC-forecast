import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def create_materials_df():
    df = pd.read_csv('data/complete.csv')
    df = df[~df['Material'].str.contains('DELETED')]
    df['Elapsed weeks'] = df['Elapsed weeks'].astype(int)

    grp = df[[
        'Material',
        'Date',
        'Quantity'
    ]].groupby(['Material', 'Date']).sum()
    grp.reset_index(inplace=True)
    grp.sort_values(by='Date')
    mats = grp['Material'].unique().tolist()

    df_dict = {}
    for mat in mats:
        df_dict[mat] = grp[grp['Material'] == mat][['Date', 'Quantity']]

    idx = pd.date_range('01-01-2014', '30-12-2016')

    for material, mat_df in df_dict.items():
        mat_df.index = pd.DatetimeIndex(mat_df['Date'])
        mat_df.drop('Date', inplace=True, axis=1)
        df_dict[material] = mat_df.reindex(idx, fill_value=0)
    return df_dict

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def reframe_normalize_dict(df_dict, lag_day):
    reframed_dict = {}
    for material, values in df_dict.items():
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = series_to_supervised(scaled, lag_day, 1)
        reframed_dict[material] = reframed
    return reframed_dict, scaler


def split(reframed_dict, lag_day):
    split_dict = {}
    for material, reframed in reframed_dict.items():
        values = reframed.values
        n_train_days = 365 * 2
        train = values[:n_train_days, :]
        test = values[n_train_days:, :]
        # split into input and outputs
        train_X, train_y = train[:, :lag_day], train[:, -1]
        test_X, test_y = test[:, :lag_day], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], lag_day, 1))
        test_X = test_X.reshape((test_X.shape[0], lag_day, 1))
        split_dict[material] = (train_X, train_y, test_X, test_y)
    return split_dict
