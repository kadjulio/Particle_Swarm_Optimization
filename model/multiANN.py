import numpy as np

class MultiANN:
    def __init__(self, X, Y, nb_layers, layers_dim):
        self.X = X
        self.Y = Y
        self.nblayers = nb_layers
        self.dims = layers_dim

        self.layers = {}


    def layers_init(self):    
        np.random.seed(1)
        for i in range(1, self.nblayers + 1, 1):
            self.layers['W' + str(i)] = np.random.randn(self.dims[i], self.dims[i - 1]) / np.sqrt(self.dims[i -1]) 
            self.layers['b' + str(i)] = np.zeros((self.dims[i], 1))                       

        for x in self.layers:
            print (x, ':', self.layers[x])            
        return

