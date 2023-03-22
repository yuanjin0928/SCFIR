import numpy as np
#import scipy.signal as signal
import matplotlib.pyplot as plt

class fir():
    
    #-----constructor-----#
    # parameter
    #   sample: sampled signal
    #   order: order of fir filter
    #   sampleRate: sampleRate of samples
    #   cutoffFreq: cutoffFreq of filter
    #   window: window type of filter
    def __init__(self):
        self.weight = None

    #-----design fir filter-----#
    # paramenter
    #   taps: number of taps
    #   cutoffReq: cutoff frequency
    #   sampleRate: sample rate
    #   window: window type. e.g. hann, hamming, 
    # return
    #   coefficients of the filter
    def filterDesign(self, taps, cutoffReq, sampleRate, window):
        firwin = signal.firwin(taps, cutoffReq, window = window, fs=sampleRate)
        self.weight = firwin
        return firwin

    #-----filter samples-----#
    # parameter
    #   sample: samples to be filtered
    def filter(self, sample, weight):
        sample = np.array(sample)
        result = signal.convolve(sample, weight, 'full')
        return result


# A = np.array([1,2,3,4,5,6,7,8,9,10])
# B = np.array([1,2,3])
# result = signal.convolve(A, B, 'full')
# print(result)


        

