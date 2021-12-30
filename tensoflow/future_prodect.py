import pandas as pd
from pandas.io import gbq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensoflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import math
from sklearn.metrics import mean_squared_error

class future_ratio:
  
  #this is the url_form
  def __init__(self , base_url):
    self.base_url = base_url
    data = gbq.read_gbq(self.base_url , project_id='herbalites')
    self.df = data.reset_index()['amount']

  #prepare our Train and Test Data and change them into dataset
  def prepare_data(self):
      self.scaler = MinMaxScaler(feature_range = (0,1))
      self.df = self.scaler.fit_transform(np.array(self.df).reshape(-1,1))
      training_size = int(len(self.df) * 0.67)
      test_size = len(self.df) - training_size
      self.Train_data , self.Test_data = self.df[0:training_size , :] , self.df[training_size: len(self.df) , :1]
      X_train , Y_train = self.creat_dataset(self.Train_data , 100)
      X_test  , Y_test  = self.creat_dataset(self.Test_data , 100)
      X_train = X_train.reshape(X_train.shape[0] , X_train.shape[1] , 1)
      X_test  = X_test.reshape(X_test.shape[0] , X_test.shape[1] , 1)
      return X_train , X_test , Y_test , Y_train

  #function that create the dataset , i call it in the last part "prepare_data"
  @staticmethod
  def creat_dataset(data_set , time_step = 1):
    data_x = []
    data_y = [] 
    for i in range(len(data_set)-time_step-1):
      data_x.append(data_set[i : (i+time_step) ,0])
      data_y.append(data_set[i + (time_step) , 0])
    return np.array(data_x) , np.array(data_y)

  #now we gonna make and train our model
  def train_model(self):
    X_train , X_test , Y_test , Y_train = self.prepare_data()
    self.model = Sequential()
    self.model.add(LSTM(50 , return_sequences = True , input_shape = (100 , 1)))
    self.model.add(LSTM(50 , return_sequences = True))
    self.model.add(LSTM(50))
    self.model.add(Dense(1))
    self.model.compile(optimizer='adam',
                  loss = 'mean_squared_error',
                  metrics = ['accuracy'])
    self.model.fit(X_train , Y_train , validation_data = (X_test , Y_test) , epochs = 100)
    train_product = self.model.predict(X_train)
    test_product  = self.model.predict(X_test)
    train_product = self.scaler.inverse_transform(train_product)
    test_product  = self.scaler.inverse_transform(test_product) 
    x_input = self.Test_data[len(self.Test_data) - 100 :].reshape(1 , -1)
    temp_input =  list(x_input)[0].tolist()
    return x_input , temp_input

  #after training and product our model , we cam use the following function to product future time sires
  def predect_future(self):
    x_input , temp_input = self.train_model()
    lat_liste  = []
    n_steps = 100
    i = 0
    for i in range(30):
      if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1 , -1)
        x_input = x_input.reshape((1 , n_steps , -1))
        y_input   = self.model.predict(x_input , verbose = 0)
        temp_input.extend(y_input[0].tolist())
        temp_input = temp_input[1:]
        lat_liste.extend(y_input.tolist())
      else :
        x_input = x_input.reshape((1, n_steps,1))
        y_input   = self.model.predict(x_input, verbose=0)
        temp_input.extend(y_input[0].tolist())
        lat_liste.extend(y_input.tolist())
    return lat_liste

  #and now after having the next 30 data , we can just show it in a graph
  def show_graph(self):
    next_future = self.predect_future()
    new  = np.arange(1,101)
    pred = np.arange(101,131)
    plt.figure(figsize = (18,6))
    plt.plot(new,self.scaler.inverse_transform(self.df[900:]))
    plt.plot(pred,self.scaler.inverse_transform(next_future))
    plt.show()

test = future_ratio("""SELECT * FROM `herbalites.braintree_herbaly_herbaly.transactions` LIMIT 1000""")
test.show_graph()