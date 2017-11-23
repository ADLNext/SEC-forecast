import numpy as np
import pandas as pd

from fbprophet import Prophet

import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)

df = pd.read_csv('data/complete.csv')
df = df[~df['Material'].str.contains('DELETED')]
df = df[~df['Quantity'] < 0]

df = df[[
    'Material',
    'Quantity',
    'Date'
]]

df['Date'] = pd.to_datetime(df['Date'])
grp = df[df['Date'].dt.year < 2016]
grp = grp.groupby('Material').sum()
top_mat = grp[[
    'Quantity'
]].sort_values(by='Quantity', ascending=False).head(50).index

f = open('prophet_log.csv', 'w')
f.write('Material, Actual, Forecast, Error\n')

for mat in top_mat:
    ts = df[df['Material'] == mat]
    print('Forecasting', mat)
    ts = ts[[
        'Quantity',
        'Date'
    ]]
    ts = ts.set_index('Date')
    ts = ts.groupby(pd.TimeGrouper(freq='D')).sum()
    ts.reset_index(inplace=True)
    ts.columns = ['ds', 'y']
    ts.fillna(0, inplace=True)

    ts_train = ts[ts['ds'].dt.year < 2016]
    ts_test = ts[ts['ds'].dt.year == 2016]

    m = Prophet()
    m.fit(ts_train)

    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    pred = forecast[forecast['ds'].dt.year == 2016]['yhat'].values
    actual = ts_test['y'].values
    error = (pred.sum() - actual.sum())/actual.sum()

    pred = pred[np.where(pred > 0)]

    f.write('%s, %d, %f, %f\n' % (
        mat.replace(',', ' '),
        actual.sum(),
        pred.sum(),
        error
    ))
