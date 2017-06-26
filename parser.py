from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply
from keras.regularizers    import l2
from keras.layers.core     import Reshape
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re


timesteps   = 200
inputs      = Input(shape=(timesteps, 128))
encoded     = LSTM(512)(inputs)
inputs_a    = inputs
inputs_a    = Dense(2048)(inputs_a)
inputs_a    = BN()(inputs_a)
a_vector    = Dense(512, activation='softmax')(Flatten()(inputs_a))
mul         = multiply([encoded, a_vector]) 
encoder     = Model(inputs, mul)

x           = RepeatVector(timesteps)(mul)
x           = Bi(LSTM(512, return_sequences=True))(x)
decoded     = TD(Dense(128, activation='softmax'))(x)

parser      = Model(inputs, decoded)
parser.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  c_i = pickle.loads( open("dataset/c_i.pkl", "rb").read() )
  xss = []
  yss = []
  with open("dataset/corpus.distinct.txt", "r") as f:
    lines = [line for line in f]
  
