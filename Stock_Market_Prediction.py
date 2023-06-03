""" Package Installation"""

# Installing yfinance
!pip install yfinance

#Importing required libraries
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf

""" Data Retrieval and Exploration"""

data = yf.download("GOOGL" , start = "2020-01-01" , interval = '1d')

data.shape

data.head(3)

# Understanding Trends with in the Data

# Sort the data points based on indexes just for confirmation 
data.sort_index(inplace = True)

# Remove any duplicate index 
data = data.loc[~data.index.duplicated(keep='first')]

data.tail(3)

# Check for missing values 
data.isnull().sum()

# Get the statistics of the data
data.describe()

import plotly.graph_objects as go

# Check the trend in Closing Values 
fig = go.Figure()

fig.add_trace(go.Scatter(x = data.index , y = data['Close'] , mode = 'lines'))
fig.update_layout(height = 500 , width = 900, 
                  xaxis_title='Date' , yaxis_title='Close')
fig.show()

# Check the trend in Volume Traded
fig = go.Figure()

fig.add_trace(go.Scatter(x = data.index , y = data['Volume'] , mode = 'lines'))
fig.update_layout(height = 500 , width = 900, 
                  xaxis_title='Date' , yaxis_title='Volume')
fig.show()

"""Data Preparation"""

from sklearn.preprocessing import MinMaxScaler 
import pickle 
from tqdm.notebook import tnrange

# Filter only required data 
data = data[['Close' , 'Volume']]
data.head(3)

# Confirm the Testing Set length 
test_length = data[(data.index >= '2023-01-01')].shape[0]

def CreateFeatures_and_Targets(data, feature_length):
    X = []
    Y = []

    for i in tnrange(len(data) - feature_length): 
        X.append(data.iloc[i : i + feature_length,:].values)
        Y.append(data["Close"].values[i+feature_length])

    X = np.array(X)
    Y = np.array(Y)

    return X , Y

X , Y = CreateFeatures_and_Targets(data , 32)

# Check the shapes
X.shape , Y.shape

Xtrain , Xtest , Ytrain , Ytest = X[:-test_length] , X[-test_length:] , Y[:-test_length] , Y[-test_length:]

# Check Training Dataset Shape 
Xtrain.shape , Ytrain.shape

# Check Testing Dataset Shape
Xtest.shape , Ytest.shape

# Create a Scaler to Scale Vectors with Multiple Dimensions 
class MultiDimensionScaler():
    def __init__(self):
        self.scalers = []

    def fit_transform(self , X):
        total_dims = X.shape[2]
        for i in range(total_dims):
            Scaler = MinMaxScaler()
            X[:, :, i] = Scaler.fit_transform(X[:, :, i])
            self.scalers.append(Scaler)
        return X

    def transform(self , X):
        for i in range(X.shape[2]):
            X[:, :, i] = self.scalers[i].transform(X[:,:,i])
        return X

Feature_Scaler = MultiDimensionScaler()
Xtrain = Feature_Scaler.fit_transform(Xtrain)
Xtest = Feature_Scaler.transform(Xtest)

Target_Scaler = MinMaxScaler()
Ytrain = Target_Scaler.fit_transform(Ytrain.reshape(-1,1))
Ytest = Target_Scaler.transform(Ytest.reshape(-1,1))

def save_object(obj , name : str):
    pickle_out = open(f"{name}.pck","wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def load_object(name : str):
    pickle_in = open(f"{name}.pck","rb")
    data = pickle.load(pickle_in)
    return data

# Save your objects for future purposes 
save_object(Feature_Scaler , "Feature_Scaler")
save_object(Target_Scaler , "Target_Scaler")

"""Model Building"""

from tensorflow.keras.callbacks import ModelCheckpoint , ReduceLROnPlateau

save_best = ModelCheckpoint("best_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25,patience=4, min_lr=0.00001,verbose = 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , LSTM , Bidirectional

model = Sequential()

model.add(Bidirectional(LSTM(512 ,return_sequences=True , recurrent_dropout=0.1, input_shape=(32, 2))))
model.add(LSTM(256 ,recurrent_dropout=0.1))
model.add(Dropout(0.3))
model.add(Dense(64 , activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(32 , activation='elu'))
model.add(Dense(1 , activation='linear'))

#optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.002)
model.compile(loss='mse', optimizer=optimizer)

history = model.fit(Xtrain, Ytrain,
            epochs=10,
            batch_size = 1,
            verbose=1,
            shuffle=False ,
            validation_data=(Xtest , Ytest),
            callbacks=[reduce_lr , save_best])

# Checking the model Structure 
model.summary()

# Load the best weights
model.load_weights("best_weights.h5")

"""Visualize prediction on Test Set"""

Predictions = model.predict(Xtest)

Predictions = Target_Scaler.inverse_transform(Predictions)
Actual = Target_Scaler.inverse_transform(Ytest)

Predictions = np.squeeze(Predictions , axis = 1)
Actual = np.squeeze(Actual , axis = 1)

# Creating Sample Test Dataframe
test_dataframe_dict = {'Actual' : list(Actual) , 'Predicted' : list(Predictions)}
test_df = pd.DataFrame.from_dict(test_dataframe_dict)

test_df.index = data.index[-test_length:]

test_df.head()

# Check the trend in Volume Traded
fig = go.Figure()

fig.add_trace(go.Scatter(x = test_df.index , y = Actual , mode = 'lines' , name='Actual'))
fig.add_trace(go.Scatter(x = test_df.index , y = Predictions , mode = 'lines' , name='Predicted'))
fig.show()

"""Visualize Prediction on whole data"""

Total_features = np.concatenate((Xtrain , Xtest) , axis = 0)

Total_Targets = np.concatenate((Ytrain , Ytest) , axis = 0)

Predictions = model.predict(Total_features)

Predictions = Target_Scaler.inverse_transform(Predictions)
Actual = Target_Scaler.inverse_transform(Total_Targets)

Predictions = np.squeeze(Predictions , axis = 1)
Actual = np.squeeze(Actual , axis = 1)

# Check the trend in Volume Traded
fig = go.Figure()

fig.add_trace(go.Scatter(x = data.index , y = Actual , mode = 'lines' , name='Actual'))
fig.add_trace(go.Scatter(x = data.index , y = Predictions , mode = 'lines' , name='Predicted'))
fig.show()
