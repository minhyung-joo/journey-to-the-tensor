import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def normalize(benchmark):
    def normalize_with_benchmark(value):
        return value / benchmark
    
    return normalize_with_benchmark

BATCH_SIZE = 100
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 1000
EPOCHS = 10
past_history = 100
future_target = 1
STEP = 1
price_data = pd.read_csv('new_prices.csv')
symbols = price_data["symbol"].drop_duplicates()
stock_data = pd.read_csv('securities.csv')
price_data.insert(0, 'GICS Sector', 'none')

stock_to_sector = dict()
for index, row in stock_data.iterrows():
    stock_to_sector[row['Ticker symbol']] = row['GICS Sector']

def assign_sector(symbol):
    return stock_to_sector[symbol]

price_data['GICS Sector'] = price_data['symbol'].map(assign_sector)
print (price_data.head())
sectors = price_data["GICS Sector"].drop_duplicates()

for sector in sectors[:1]:
    price_hist = price_data.loc[price_data['GICS Sector'] == sector]
    compiled_x = np.zeros((1, 100, 3))
    compiled_y = np.zeros((1,))
    for symbol in symbols:
        stock_price_hist = price_hist.loc[price_hist['symbol'] == symbol]
        if len(stock_price_hist) > 0:
            stock_price_hist.drop_duplicates("date")
            features = stock_price_hist[["close", "low", "high"]]
            features.index = stock_price_hist['date']
            
            benchmark = features.iat[0, 0]
            normalizer = normalize(benchmark)
            features["close"] = features["close"].map(normalizer)
            features["low"] = features["low"].map(normalizer)
            features["high"] = features["high"].map(normalizer)
            dataset = features.values

            x, y = multivariate_data(dataset, dataset[:,0], 0, None, past_history, future_target, STEP, True)
            compiled_x = np.concatenate((compiled_x, x))
            compiled_y = np.concatenate((compiled_y, y))
    
    train_data_single = tf.data.Dataset.from_tensor_slices((compiled_x, compiled_y))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    input_shape = None
    
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))
    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    history = single_step_model.fit(train_data_single, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL)

    single_step_model.save_weights('./financial_weights')
"""
for symbol in symbols[:1]:
    price_hist = price_data.loc[price_data['symbol'] == symbol]
    price_hist.drop_duplicates("date")
    features = price_hist[["close", "low", "high"]]
    features.index = price_hist['date']
    
    benchmark = features.iat[0, 0]
    normalizer = normalize(benchmark)
    features["close"] = features["close"].map(normalizer)
    features["low"] = features["low"].map(normalizer)
    features["high"] = features["high"].map(normalizer)
    dataset = features.values

    x, y = multivariate_data(dataset, dataset[:,0], 0, None, past_history, future_target, STEP, True)

    train_data_single = tf.data.Dataset.from_tensor_slices((x, y))
    train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    input_shape = None
    
    single_step_model = tf.keras.models.Sequential()
    single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x.shape[-2:]))
    single_step_model.add(tf.keras.layers.Dense(1))
    single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

    history = single_step_model.fit(train_data_single, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL)
    for x, y in train_data_single.take(1):
        predictions = single_step_model.predict(x)
        print (predictions * benchmark)
"""