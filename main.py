import os

import model as ml
import functions.activitions as act

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
feature_set, labels = datasets.make_moons(1, noise=0.10)

def file_to_dataset(file):
    text_file = open(file, "r")
    lines = text_file.readlines()
    text_file.close()
    X = []
    Y = []
    for line in lines:
        val = line.split()
        X.append(float(val[0]))
        Y.append(float(val[1]))
    return X, Y

path = "Data/"
files = os.listdir(path)

for datafile in files:
    X, Y = file_to_dataset("Data/" + datafile)

    ann = ml.MultiANN(X, Y)
    ann.add_layer(ml.Layer(1, 10, act.Sigmoid))
    ann.add_layer(ml.Layer(10, 4, act.Sigmoid))
    ann.add_layer(ml.Layer(4, 1, act.Sigmoid))
    ann.train(epochs=1)
    exit(0) #to kill