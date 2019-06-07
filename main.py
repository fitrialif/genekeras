from genekeras import *

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import GaussianNoise, Activation, Dropout

import numpy as np

def create_model():
    model = Sequential()
    model.add(Dense(64, input_shape=(STATE_SIZE, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(ACTION_SIZE, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model

def create_model_special():
    model = Sequential()
    model.add(Dense(64, input_shape=(STATE_SIZE, ), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(ACTION_SIZE))
    model.add(GaussianNoise(1.0))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])

    return model


# Main Program
STATE_SIZE = 5
ACTION_SIZE = 3

mom = create_model_special()
dad = create_model_special()

gk = GeneKeras(load_compiled = True)
gk.set_parents(mom, dad)
gk.set_param(crossover_enabled = False, mutation_enabled = False, mutation_prob = 0.1, mutation_rate = 0.5)

child = gk.get_child()

print( dad.predict(np.array([[1, 1, 1, 1, 1]])) )
print( mom.predict(np.array([[1, 1, 1, 1, 1]])) )
print( child.predict(np.array([[1, 1, 1, 1, 1]])) )