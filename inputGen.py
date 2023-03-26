import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
 

def uniform(num,times):
    rng = default_rng()
    return rng.uniform(-1,1, (num, times))

