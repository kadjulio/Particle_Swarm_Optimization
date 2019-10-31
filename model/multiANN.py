import numpy as np

class MultiANN:
    def __init__(self, X, Y, nb_layers, layers_dim, activation_functions):
        self.X = X
        self.Y = Y
        self.nblayers = nb_layers
        self.dims = layers_dim
        self.layers = {}
        self.act_fcts = activation_functions
        
    def layers_init(self):    
        np.random.seed(42)
        for i in range(1, self.nblayers + 1, 1):
            self.layers['W' + str(i)] = np.random.randn(self.dims[i - 1], self.dims[i]) / np.sqrt(self.dims[i]) 

        for x in self.layers:
            print (x, ':', self.layers[x])            
        return

    def forward(self):    
        for raw in self.X:
            Z = np.dot(raw, self.layers['W1'])
            act_val = self.act_fcts[0](Z)
            print("\n\nlayers n°1: " + str(act_val))

            for i, act in zip(range(2, self.nblayers + 1, 1), self.act_fcts[1:]):
                Z = np.dot(act_val, self.layers['W' + str(i)]) 
                act_val = act(Z)
                print("layers n°" + str(i) + ": " + str(act_val))
        
        self.Yo=act_val
        return self.Yo

    def train(self, epochs):
        for epoch in range(epochs):
            self.forward()
