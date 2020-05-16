"""
        stock price prediction
"""

# data preprocessing
import os

os.chdir("D:\machine learning\Recurrent_Neural_Networks/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Google_Stock_Price_Train.csv")

#values will comvert dataframe into numpy array
training_set=data.iloc[:,1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train_scaler_set=sc.fit_transform(training_set)

# data structure with 60 timestep and 1 output
# means our network predict next 1 value using previous 60 steps
# predict(t-60,t-59 ... t) -> t+1
x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(train_scaler_set[i-60:i])
    y_train.append(train_scaler_set[i])

x_train,y_train=np.array(x_train),np.array(y_train)

#reshaping data to adding more dimensionality in data for better predictions

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))



#building RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout

#initialize RNN

model= Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50)) # remove return_sequence because it is a last layer
model.add(Dropout(0.2))

model.add(Dense(1))


model.compile(optimizer='RMSprop',loss='mean_squared_error')


model.fit(x_train,y_train,epochs=100,batch_size=32)


# predict 

test=pd.read_csv("Google_Stock_Price_Test.csv")

test_set=test.iloc[:,1:2].values


total=pd.concat((data['Open'],test['Open']),axis=0)

inputs=total[len(total)-len(test)-60:].values
inputs=inputs.reshape(-1,1)

inputs=sc.transform(inputs)


x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i])
    #y_train.append(train_scaler_set[i])

x_test=np.array(x_test)

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predict=model.predict(x_test)

# inverse it to it's real form
predict=sc.inverse_transform(predict)


#visualize the price

plt.plot(test['Open'])

plt.plot(predict)