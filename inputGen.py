import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

def random(para, num, times):
    rng = default_rng()
    return (para[1]-para[0])*rng.random(num, times) + para[0]

def uniform(para, num,times):
    rng = default_rng()
    return rng.uniform(para[0], para[1], (num, times))

def nomal(para, num, times):
    rng = default_rng()
    return rng.normal(para[0], para[1], (num, times))