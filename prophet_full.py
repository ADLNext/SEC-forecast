import numpy as np
import pandas as pd

from fbprophet import Prophet

df = pd.read_csv('data/complete.csv')
df = df[~df['Material'].str.contains('DELETED')]
df = df[~df['Quantity'] < 0]

df = df[[
    'Material',
    'Quantity',
    'Date',
    'Region',
    'Category'
]]

grp = df.groupby('Material').sum()
top_mat = grp[[
    'Quantity'
]].sort_values(by='Quantity', ascending=False).head(50).index

f = open('prophet_log_full.csv', 'w')
f.write('Material, Region, Category, Actual, Forecast, Error\n')

for mat in top_mat:
    ts_mat = df[df['Material'] == mat]
    print('Forecasting', mat)
    for region in df['Region'].unique():
        ts_reg = ts_mat[ts_mat['Region'] == region]
        for cat in df['Category'].unique():
            ts = ts_reg[ts_reg['Category'] == cat]
            cap = ts['Quantity'].max()*1.5
            ts = ts[[
                'Quantity',
                'Date'
            ]]
            ts = ts.set_index('Date')
            ts.index = pd.to_datetime(ts.index)
            ts = ts.groupby(pd.TimeGrouper(freq='D')).sum()
            ts.reset_index(inplace=True)
            ts.columns = ['ds', 'y']
            ts.fillna(0, inplace=True)

            ts_train = ts[ts['ds'].dt.year < 2016]
            ts_test = ts[ts['ds'].dt.year == 2016]

            m = Prophet()
            try:
                m.fit(ts_train);
            except ValueError:
                f.write('%s, %s, %s, %d, nan, nan\n' % (
                    mat.replace(',', ' '),
                    region,
                    cat,
                    ts_test['y'].values.sum()
                ))
                continue

            future = m.make_future_dataframe(periods=365)
            forecast = m.predict(future)

            pred = forecast[forecast['ds'].dt.year == 2016]['yhat'].values
            pred = pred[np.where(pred > 0)]
            actual = ts_test['y'].values
            error = (pred.sum() - actual.sum())/actual.sum()

            f.write('%s, %s, %s, %d, %f, %f\n' % (
                mat.replace(',', ' '),
                region,
                cat,
                actual.sum(),
                pred.sum(),
                error
            ))
