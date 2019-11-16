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
        X.append(float(val[0]))
        Y.append(float(val[1]))
    return X, Y

path = "Data/"
files = os.listdir(path)

for datafile in files:
    X, Y = file_to_dataset("Data/" + datafile)

    for config in configs:
        print(config["shape"])
        ann = ml.MultiANN(X, Y)
        for idx, (shp, act) in enumerate(zip(config["shape"][:-1], config["activations"])):
            ann.add_layer(ml.Layer(config["shape"][idx], config["shape"][idx + 1], act))
        ann.train(config)
    
    exit(0) #to kill