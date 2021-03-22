# Imports

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from sklearn.model_selection import train_test_split



# Load Data
print('\nLoading data...\n')
data = pd.read_csv("data/weatherHistory.csv")
temp_data = data.iloc[:,3:4].values #temperature column only
temp_data




# Encode the data
print("\nEncoding Data with MinMax Scaler...\n")
mms = MinMaxScaler(feature_range = (0,1))
temp_data = mms.fit_transform(temp_data)





# Train data (OPTION 1)
print("\nInitializing....\n")
x_train = []
y_train = []
n_future = 7     #7 days temperature forecast
n_past = 30      #Past 30 days 

for i in range(0,len(temp_data)-n_past-n_future+1):
    x_train.append(temp_data[i : i + n_past , 0])     
    y_train.append(temp_data[i + n_past : i + n_past + n_future , 0 ])

x_train , y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0] , x_train.shape[1], 1) )






# LSTM Model Defninition
print("\nDefining and Building LSTM Models...\n")
model = Sequential()
model.add(Bidirectional(LSTM(units=50, 
                             activation= "sigmoid", 
                             return_sequences=True, 
                             input_shape = (x_train.shape[1], 1))))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units= 50))
model.add(Dense(units = n_future , activation="sigmoid"))  







# Train Model
print("\nTraining Model \n")
model.compile(optimizer='adam', 
              loss = 'mean_squared_error', 
              metrics='accuracy')
model.fit(x_train, 
          y_train, 
          epochs=100, 
          batch_size=128)