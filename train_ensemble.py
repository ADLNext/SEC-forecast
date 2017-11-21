import os.path

import support

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

# how many days in the past is the model looking at?
# I need to decide this parameter now because it will change the way in which I prepare the data for the model
lag_day = 30

# Step 1: creating a different dataframe for each material
# The df contains daily consumption for each material from Jan 1st 2014 to Dec 31st 2016
df_dict = support.create_materials_df()

# Step 2: reframing and normalizing
# reframing is basically creating from the original sequence a bunch of lag-day-long sequences
# normalization is squashing all the data between 0 and 1, this is required to properly train the model
reframed_dict = support.reframe_normalize_dict(df_dict, lag_day)
# deleting the old dictionary to save RAM and help pandas keep everything in-memory
del df_dict

# Step 3: splitting into training and testing set
split_dict = support.split(reframed_dict, lag_day)
# deleting the old dictionary to save RAM and help pandas keep everything in-memory
del reframed_dict

# Step 4: training the models and storing them on disk for future evaluation
for material, sets in split_dict.items():
    # the file name for the model
    fname = 'models/' + material.replace('/', '') + '.h5'
    # if the file exists already I can skip this material and move on to the next one
    if os.path.isfile(fname):
        print('Skipping', material)
        continue

    # fetching training and testing sets
    train_X = sets[0]
    train_y = sets[1]
    test_X = sets[2]
    test_y = sets[3]

    # design network
    model = Sequential()
    model.add(LSTM(
        50, return_sequences=True,
        input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='rmsprop')

    print('Fitting', material, ' model')

    # fit network
    history = model.fit(
        train_X,
        train_y,
        epochs=50,
        batch_size=72,
        validation_data=(test_X, test_y),
        verbose=0,
        shuffle=False)
    loss_log = history.history['loss'][-1]
    try:
        val_loss_log = history.history['val_loss'][-1]
    except KeyError:
        val_loss_log = 'nan'
    print('loss:', loss_log, '\t val_loss:', val_loss_log)
    model.save(fname)
