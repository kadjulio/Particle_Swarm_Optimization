import numpy as np
from .layer import Layer

class MultiANN:
    shape = []
    vector_weights = None

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.layers = []
        self.shape.append(1) # append nb of inputs, here 1

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        self.shape.append(layer.w_shape[1])

    def feedforward(self):    
        for raw in self.X:
            act_val = self.layers[0].forward(raw)
            print("\n\nlayer 0: " + str(act_val))

            for idx, layer in enumerate(self.layers[1:]): 
                act_val = layer.forward(act_val)
                print("layer %d: %s" % ( idx + 1, str(act_val)))
        
        self.Yo=act_val
        print(self.Yo)
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
        for epoch in range(epochs):
            self.feedforward()
