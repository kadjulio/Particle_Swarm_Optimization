import os

import model as ml
import functions.activitions as act
from functions.cost import mse

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from test import configs

def file_to_dataset(file):
    text_file = open(file, "r")
    lines = text_file.readlines()
    text_file.close()
    X = []
    Y = []
    for line in lines:
        val = line.split()
        if len(val) == 2:
            X.append(float(val[0]))
            Y.append(float(val[1]))
        elif len(val) == 3:
            X.append([float(val[0]), float(val[1])])
            Y.append(float(val[2]))
    return X, Y

path = "Data/"
files = os.listdir(path)

for datafile in files:
    print(datafile)
    X, Y = file_to_dataset("Data/" + datafile)
    X_shape = 1 if len(np.asarray(X).shape) == 1 else np.asarray(X).shape[1]
    best_config = None
    best_mse = 1
    for indx, config in enumerate(configs):
        ann = ml.MultiANN(X, Y)
        ann.add_layer(ml.Layer(X_shape, config["shape"][0], config["activations"][0]))
        for idx, (shp, act) in enumerate(zip(config["shape"][:-1], config["activations"][1:])):
            ann.add_layer(ml.Layer(config["shape"][idx], config["shape"][idx + 1], act))
        mse = ann.train(config)
        if mse < best_mse:
            config["idx"] = indx + 1
            best_config = config
            best_mse = mse
    print("\tBEST CONFIG n°%d with %.3f loss" % (best_config["idx"], best_mse))
    
    # exit()
    # TODO refacto code