import support

import numpy as np

from sklearn.metrics import mean_squared_error
from keras.models import load_model

# recreating all the dictionaries, see comment in train_ensemble.py

lag_day = 30

df_dict = support.create_materials_df()

reframed_dict, scaler = support.reframe_normalize_dict(df_dict, lag_day)
del df_dict

split_dict = support.split(reframed_dict, lag_day)
del reframed_dict

# csv file to log predictions vs actuals
f = open('pred_log.csv', 'w')

rmse_coll = np.array([])
for material, sets in split_dict.items():
    # loading the trained model from file
    fname = 'models/' + material.replace('/', '') + '.h5'
    print('Loading', material)
    model = load_model(fname)

    # fetching the test sets (year 2016)
    test_X = split_dict[material][2]
    test_y = split_dict[material][3]
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], lag_day))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -1:]), axis=1)
    inv_yhat = np.nan_to_num(inv_yhat)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    # log predictions and actuals
    f.write('%s, %f\n' % (material.replace(',', ' '), rmse))
    f.write('Actuals')
    for act in inv_y:
        f.write(', %f' % act)
    f.write('\n')
    f.write('Predictions')
    for pred in inv_yhat:
        f.write(', %f' % pred)
    f.write('\n')
    del model

# save file with all losses to perform further analysis
np.save('rmse.npy', rmse_coll)
