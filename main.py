import model as ml
import functions.activitions as act

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
feature_set, labels = datasets.make_moons(1, noise=0.10)
Y = np.random.randn(len(feature_set[0]))

ann = ml.MultiANN(feature_set, Y)
ann.add_layer(ml.Layer(len(feature_set[0]), 10, act.Sigmoid))
ann.add_layer(ml.Layer(10, 4, act.Sigmoid))
ann.add_layer(ml.Layer(4, 1, act.Sigmoid))
ann.train(epochs=1)
