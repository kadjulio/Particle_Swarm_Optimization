from typing import List
import numpy as np


def Sigmoid(Z: float) -> float:
    return 1 / (1 + np.exp(-Z))

def Relu(Z: float) -> float:
    return np.maximum(Z, 0, Z)

def Null(Z: float) -> float:
    return 0

def Tanh(Z: float) -> float:
    #return np.sinh(Z)/np.cosh(Z)
    return np.tanh(Z)

def Cos(Z: float) -> float:
    return np.cos(Z)

def Gaussian(Z: float) -> float:
    return np.exp(-(Z^2/2))

def Softmax(x: List(float)) -> List(float):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()