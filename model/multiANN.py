import numpy as np
from .layer import Layer

class MultiANN:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
    
    
    def feedforward(self):    
        for raw in self.X:
            act_val = self.layers[0].forward(raw)
            print("\n\nlayer 0: " + str(act_val))

            for idx, layer in enumerate(self.layers[1:]): 
                act_val = layer.forward(act_val)
                print("layer %d: %s" % ( idx + 1, str(act_val)))
        
        self.Yo=act_val
        return self.Yo

    def train(self, epochs):
        for epoch in range(epochs):
            self.feedforward()
