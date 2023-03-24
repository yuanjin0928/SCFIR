import os
import numpy as np
import matplotlib.pyplot as plt
import inputGen
import weightGen
import test
import FIRRun

"""generate input"""
power = 5
num, times = 2**power, 10
samples = inputGen.uniform(num, times)

"""generate weight"""
weight = weightGen.itentical(1/32, num)

"""test effect of sc's length on the precision of filter"""
minLen = 8
maxLen = 20
CVTError = test.Test_SCLen('CVT', minLen, maxLen, samples, weight)
HWAError = test.Test_SCLen('HWA', minLen, maxLen, samples, weight)
MWAError = test.Test_SCLen('MWA', minLen, maxLen, samples, weight)

"""compare result"""
xaxis = list(range(minLen, maxLen+1))
plt.plot(xaxis, CVTError, 'b-o', xaxis, HWAError, 'g-o', xaxis, MWAError, 'r-o')
for i in range(len(xaxis)):
    plt.text(xaxis[i], CVTError[i], str(round(CVTError[i], 5)))
    plt.text(xaxis[i], HWAError[i], str(round(HWAError[i], 5)))
    plt.text(xaxis[i], MWAError[i], str(round(MWAError[i], 5)))
#plt.plot(xaxis, error, 'r-o')
plt.show()