import numpy as np


def f1(x):
    return x * np.sin(x)


def f2(x1, x2):
    return (x1 * np.sin(x1*0.05+np.pi) + x2 * np.cos(x2*0.05)) * 0.1
