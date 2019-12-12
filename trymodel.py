import tensorflow as tf
import pandas as pd

def normalize(benchmark):
    def normalize_with_benchmark(value):
        return value / benchmark
    
    return normalize_with_benchmark

amex_data = pd.read_csv('CME.csv')
features = amex_data[["Close", "Low", "High"]]
features.index = amex_data['Date']
print (features.values[-100:])
benchmark = features.iat[0, 0]
normalizer = normalize(benchmark)
features["Close"] = features["Close"].map(normalizer)
features["Low"] = features["Low"].map(normalizer)
features["High"] = features["High"].map(normalizer)
dataset = features.values
x = dataset[-100:].reshape((1, 100, 3))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32, input_shape=x.shape[-2:]))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

model.load_weights('./financial_weights')
prediction = model.predict(x)
print (prediction * benchmark)