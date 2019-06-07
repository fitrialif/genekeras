import keras
from keras.models import Model
from keras.layers import Dense
from keras.models import clone_model, load_model
import numpy as np
import os

class GeneKeras:

    def __init__(self, load_compiled = False):
        self.model_1 = None
        self.model_2 = None

        self.load_compiled = load_compiled

        self.set_param()


    def set_param(self, crossover_enabled = True, mutation_enabled = True, mutation_prob = 0.1, mutation_rate = 0.5):
        self.crossover_enabled = crossover_enabled
        self.mutation_enabled = mutation_enabled
        self.mutation_prob = mutation_prob
        self.mutation_rate = mutation_rate


    def set_parents(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2

        assert self.check_models_shape(), "Different model shape!"


    def get_child(self):
        if (not self.load_compiled):
            self.model_3 = clone_model(self.model_1)
            self.model_3.set_weights(self.model_1.get_weights())
        else:
            # Is this the best solution? Use an external file? I doubt about it tbh
            self.model_1.save("temp.h5")
            self.model_3 = load_model("temp.h5")
            os.remove("temp.h5")

        if (self.crossover_enabled):
            self.crossover()

        if (self.mutation_enabled):
            self.mutation()

        return self.model_3


    def get_children(self, number):
        children = []
        for i in range (number):
            children.append(get_child())
        return children


    def crossover(self):
        available_layers_number = 0

        layer_number = len(self.model_3.layers)

        for i in range (0, layer_number):
            if (isinstance(self.model_3.layers[i], Dense)):
                available_layers_number += 1

        cut_layer = np.random.random_integers(0, available_layers_number)

        j = 0
        for i in range(0, layer_number):
            if (isinstance(self.model_3.layers[i], Dense)):
                if(j < cut_layer):
                    self.model_3.layers[i].set_weights(self.model_1.layers[i].get_weights())
                else:
                    self.model_3.layers[i].set_weights(self.model_2.layers[i].get_weights())
                j += 1


    def mutation(self):
        layer_number = len(self.model_3.layers)
        for i in range (0, layer_number):
            if (isinstance(self.model_3.layers[i], Dense)):
                weights, biases = self.model_3.layers[i].get_weights()
                for k in range (len(weights)):
                    if np.random.uniform() < self.mutation_prob:
                        weights[k] += (np.random.uniform(-self.mutation_rate, self.mutation_rate))
                for k in range (len(biases)):
                    if np.random.uniform() < self.mutation_prob:
                        biases[k] += (np.random.uniform(-self.mutation_rate, self.mutation_rate))
                self.model_3.layers[i].set_weights([weights, biases])


    def check_models_shape(self):
        layer_numbers_1 = len(self.model_1.layers)
        layer_numbers_2 = len(self.model_2.layers)

        if(layer_numbers_1 != layer_numbers_2):
            return False

        for n in range (layer_numbers_1):
            if (isinstance(self.model_1.layers[n], Dense) and isinstance(self.model_1.layers[n], Dense)):
                if (self.model_1.layers[n].get_weights()[0].shape != self.model_2.layers[n].get_weights()[0].shape):
                    return False

        return True
