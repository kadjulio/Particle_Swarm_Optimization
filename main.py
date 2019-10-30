import model as ml
import functions.activitions as activations

ann = ml.MultiANN(1, 2, 2, [2, 2, 2])
ann.layers_init()
ann.forward([activations.Relu, activations.Sigmoid])