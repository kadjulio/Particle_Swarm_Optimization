import model as ml
import functions.activitions as act

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
feature_set, labels = datasets.make_moons(3, noise=0.10)
Y = np.random.randn(len(feature_set[0]))
nb_layers = 2
layers_dim = [len(feature_set[0]), 4, 1] #first value corresponds to input dimension
activation_functions = [act.Sigmoid, act.Sigmoid]

ann = ml.MultiANN(feature_set, Y, nb_layers, layers_dim, activation_functions)

ann.layers_init()
ann.train(epochs=2)
