import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

st.title('Stock market forecasting using Deep Learning')

st.warning('We utilized an LSTM model to predict this data, but it should not be employed for actual transactions. This project is undertaken during the summer with the primary aim of delving into Deep Learning techniques. ', icon="⚠️")

# Define the list of companies and their corresponding symbols
companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL"
}

# User selects a company
company = st.radio(
    "Choose a company",
    list(companies.keys())
)

# Load stock data
start = '2010-01-01'
end = '2019-12-31'
symbol = companies[company]
df = yf.download(symbol, start=start, end=end)
df = df['Close'].values

df = df.reshape(-1, 1)

dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])



scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)


dataset_test = scaler.transform(dataset_test)

def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i-50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=1, batch_size=32)
model.save('stock_prediction.h5')

model = load_model('stock_prediction.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create and display the plot
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')
ax.plot(y_test_scaled, color='red', label='Original price')
ax.plot(predictions, color='cyan', label='Predicted price')
ax.legend()

# Cache the plot for the selected company
st.pyplot(fig)

# Save the cached plot in a file (optional)
cached_plot_filename = f"{company}_plot.png"
fig.savefig(cached_plot_filename)
st.text(f"Cached plot saved as {cached_plot_filename}")
