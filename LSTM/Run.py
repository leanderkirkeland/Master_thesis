from cProfile import label
from turtle import color
import tensorflow as tf
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from Code.LSTM.Model import LSTM64x64
import xarray as xr
import matplotlib.pyplot as plt

train_x = []
train_y = []
test_x = []
test_y = []

for i in range(180):
    data = xr.open_dataset(f"Datasets/64x64test{i}.nc")
    train_x.append(data["n"])
    train_y.append(data["blob_labels"])

for i in range(20):
    data = xr.open_dataset(f"Datasets/64x64test{i+180}.nc").to_array()
    test_x.append(data[0])
    test_y.append(data[1])

train_x = np.moveaxis(np.array(train_x).reshape(180,64,64,200,1),3,1)
train_y = np.moveaxis(np.array(train_y).reshape(180,64,64,200,1),3,1)
test_x = np.moveaxis(np.array(test_x).reshape(20,64,64,200,1),3,1)
test_y = np.moveaxis(np.array(test_y).reshape(20,64,64,200,1),3,1)

model = LSTM64x64()

weights = np.where(train_y == 0, 0.01, 0.5)

model.fit(train_x, train_y, epochs=20, batch_size=4, sample_weight=weights)

model.save("LSTMtest")

model = load_model("LSTMtest")

x = model.predict(test_x[0:2])

x = np.argmax(x,-1)

plt.imshow(x[0,50,:,:])
plt.show()

plt.imshow(test_y[0,50,:,:])
plt.show()

model.summary()