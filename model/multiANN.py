from .layer import Layer
from functions.cost import mse, loss_ann
from PSO.pso import PSO

import numpy as np
import matplotlib.pyplot as plt


def autobuild_ann(weights, shape, activations, X, Y):
    model = MultiANN(X, Y)
    for idx, (shp, act) in enumerate(zip(shape[:-1], activations)):
        model.add_layer(Layer(shape[idx], shape[idx + 1], act))
    model.vector_weights = np.asarray(weights)
    model.update_weights()
    return model

def func1(weights, shape, activations, X, Y):
    model = autobuild_ann(weights, shape, activations, X, Y)
    Y_pred = []
    for input_x, y_true in zip(X, Y):
            y_pred = model.feedforward(input_x)[0][0]
            Y_pred.append(y_pred)
    return loss_ann(Y, Y_pred)

class MultiANN:
    
    vector_weights = None

    def __init__(self, X, Y):
        self.shape = []
        self.activations = []
        self.X = X
        self.Y = Y
        self.layers = []
        self.shape.append(1) # append nb of inputs, here 1
    
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        self.shape.append(layer.w_shape[1])
        self.activations.append(layer.activation)
    
    def feedforward(self, raw):    
        act_val = self.layers[0].forward(raw)
        # print("\n\nlayer 0: " + str(act_val))

        for idx, layer in enumerate(self.layers[1:]): 
            act_val = layer.forward(act_val)
            # print("layer %d: %s" % ( idx + 1, str(act_val)))
        
        self.Yo=act_val
        # print(self.Yo)
        return self.Yo

    def weights_to_vector(self, weights):
        w = np.asarray([])
        for i in range(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        return w

    def vector_to_weights(self, vector, shape):
        weights = []
        idx = 0
        for i in range(len(shape)-1):
            r = shape[i]
            c = shape[i + 1]
            idx_min = idx
            idx_max = idx + r*c
            W = vector[idx_min:idx_max].reshape(r,c)
            weights.append(W)
            idx = idx_max
        return weights
    
    def vectorize_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(layer.weights)
        self.vector_weights = self.weights_to_vector(weights)
    
    def update_weights(self):
        weights = self.vector_to_weights(self.vector_weights, self.shape)
        for layer, n_weight in zip(self.layers, weights):
            layer.weights = n_weight

    def train(self, epochs):
        losses = []
        losses_pso = []
        for input_x, y_true in zip(self.X, self.Y):
            y_pred = self.feedforward(input_x)
            loss = mse(y_pred[0][0], y_true)
            losses.append(loss)
            self.vectorize_weights()
        # print("WEIGHTS: ", self.layers[0].weights)
        # print("SHAPE: ", self.shape)
        # print("ACTIVATION :", self.activations)
        print(self.vector_weights)
        self.vector_weights = np.asarray(PSO(func1, self.vector_weights, self.shape, self.activations, self.X, self.Y, 0, 400, 10).get_pos_best_g())
        print(self.vector_weights)
        self.update_weights()
        print("NEW WEIGHTS: ", self.layers[0].weights)
        for input_x, y_true in zip(self.X, self.Y):
            y_pso_pred = self.feedforward(input_x)
            loss_pso = mse(y_pso_pred[0][0], y_true)
            losses_pso.append(loss_pso)
        print("loss pso -->", loss_pso)
        print("loss -->", loss)
        print("Global loss ----> ", sum(losses) / len (losses))
        print("Global loss pso ----> ", sum(losses_pso) / len (losses_pso))
        plt.plot(losses)
        plt.plot(losses_pso)
        plt.show()