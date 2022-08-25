import tensorflow as tf
import numpy as np 
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import xarray as xr


def LSTM64x64(input_size = (200,64,64,1)):
  input = Input(input_size) 
  LSTM1 = ConvLSTM2D(16, 3, return_sequences=True, padding="same",activation="relu")(input)

  Pool1 = TimeDistributed(MaxPooling2D((2,2), strides=(2,2)))(LSTM1)

  LSTM4 = ConvLSTM2D(16, 3, return_sequences=True, padding="same",activation="relu")(Pool1)


  Pool4 = TimeDistributed(UpSampling2D((2,2)))(LSTM4)
  CONV4 = Conv2D(16, 3 , padding = "same",activation="relu")(Pool4)
  output = Conv2D(40, 3 , padding = "same", activation = "softmax")(CONV4)

  model = Model(inputs = input, outputs = [output])
  model.compile(loss='sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics=["accuracy"], sample_weight_mode='temporal')
  return model
