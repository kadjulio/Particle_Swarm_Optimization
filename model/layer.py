import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        self.w_shape = (input_size, output_size)
        self.weights = np.random.randn(self.w_shape[0], self.w_shape[1]) / np.sqrt(self.w_shape[1])
        
        self.b_shape = (1, output_size)
        self.bias = np.random.randn(1, self.b_shape[1])
        
        # print ("\n%d neurons with %d inputs" % (self.w_shape[1], self.w_shape[0]))
        # print(self.weights)

    def forward(self, X):
        Z = np.dot(X, self.weights)# + self.bias
        act_val = self.activation(Z)
        return act_val
            