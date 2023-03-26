import numpy as np
from scipy import signal

def itentical(value, num):
    return np.zeros(num) + value


def lowPassFir(cutFreq, order):
    window = 'hamming'
    coefficients = signal.firwin(order+1, cutFreq, window=window, fs=360)

    return coefficients