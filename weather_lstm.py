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

data = pd.read_csv("weatherHistory.csv")
dates = pd.DataFrame(data.iloc[:,0:1].values)
temp_data = data.iloc[:,3:4].values #temperature column only
temp_data
print(type(dates))


# Optional Filter

filter_active = 0
if filter_active == True:
  temp_data = medfilt(temp_data, 3)
  temp_data = gaussian_filter1d(temp_data, 1.2)


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

model = Sequential()
model.add(Bidirectional(LSTM(units=50, 
                             activation= "relu", 
                             return_sequences=True, 
                             input_shape = (x_train.shape[1], 1))))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units= 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units= 50))
model.add(Dense(units = n_future , activation="relu"))  


# Extra Model Functionality

# Model Checkpoint 
checkpoint_filepath = '/checkpoints'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=True)


# Early Stopping
stopping = EarlyStopping(monitor='loss', patience=5)



# Train Model

model.compile(optimizer='adam', 
              loss = 'mean_squared_error', 
              metrics='accuracy')
history = model.fit(x_train, 
          y_train, 
          epochs=5, 
          batch_size=100,
          verbose=1,
          callbacks=[checkpoint, stopping]
          )



# Visualize Metrics

# Loss
def visualize_loss(history, title):
    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "ro--", label="Training loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

# Accuracy
def visualize_acc(history, title):
    acc = history.history["accuracy"]
    epochs = range(len(acc))
    plt.figure()
    plt.plot(epochs, acc, "ro--", label="Training loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()



visualize_loss(history, "Training Loss")
visualize_acc(history, "Training Accuracy")
