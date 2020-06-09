#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:06:40 2020

@author: jaisi8631
"""
# --------------------------
# PART 1: IMPORTS
# --------------------------
# data management imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# machine learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from sklearn.externals import joblib


# --------------------------
# PART 2: DATA RETRIEVAL
# --------------------------
# import file containing historical stock price data
df = pd.read_csv('stock_data.csv')
df.head()

# manipulate index of data to store data in formatted style
df['date'] = pd.to_datetime(df.date,format='%Y-%m-%d')
df.index = df['date']

# create new dataframe with date and close values only
data = df.sort_index(ascending = True, axis = 0)
processed_data = pd.DataFrame(index = range(0, len(df)), columns = ['date', 'close'])
for i in range(0, len(data)):
    processed_data['date'][i] = data['date'][i]
    processed_data['close'][i] = data['close'][i]
    
# setting index to be numerical values
processed_data.index = processed_data.date
processed_data.drop('date', axis = 1, inplace = True)


# --------------------------
# PART 3: DATASET GENERATION
# --------------------------
# initialize figures and data
scaler = MinMaxScaler(feature_range=(0, 1))
joblib.dump(scaler, "scaler.save") 

# creating final dataset
dataset = processed_data.values

# initiating split for train and test dataset
size = int(len(dataset) * 0.8)
train = dataset[0:size, :]
test = dataset[size:, :]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# each training instance uses data from the previous 30 days
x_train = []
y_train = []
for i in range(30, len(train) - 30):
    x_train.append(scaled_data[i - 30:i, 0])
    y_train.append(scaled_data[i, 0])

# reshaping training arrays
x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
y_train = np.array(y_train)


# --------------------------
# PART 4: MODEL GENERATION
# --------------------------
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units = 30, return_sequences = True, 
               input_shape = (x_train.shape[1], 1)))
model.add(LSTM(units = 30))
model.add(Dense(10))
model.add(Dense(1))


# --------------------------
# PART 5: MODEL TRAINING
# --------------------------
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1, batch_size = 1)
model.save('stock-predictor.h5')


# --------------------------
# PART 6: TEST DATA CREATION
# --------------------------
# create and scale test data using previous stock prices from previous 30 days
inputs = processed_data[len(processed_data) - len(test) - 30:].values
inputs = inputs.reshape(-1, 1)
inputs  = scaler.transform(inputs)

# format testing dataset
x_test = []
for i in range(30, inputs.shape[0]):
    x_test.append(inputs[i-30:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# --------------------------
# PART 7: TEST PREDICTION
# --------------------------
# load saved model
model = load_model('stock-predictor.h5')

# predict and inverse scale predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# generate and output absolute error
mae = mean_absolute_error(test, predictions)
mae = round(mae, 2)
print("The average error of the model is: $" + str(mae))


# --------------------------
# PART 7: TEST RESULTS
# --------------------------
# create templates for lines
train_data = processed_data[:size]
test_data = processed_data[size:]
test_data['predictions'] = predictions

plt.rcParams["figure.figsize"] = (20, 10)
plt.plot(train_data['close'])
plt.plot(test_data[['close','predictions']])
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Stock Price Predictions')
plt.savefig('results.png')