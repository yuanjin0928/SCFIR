import os
import numpy as np
import matplotlib.pyplot as plt
import inputGen
import weightGen
import test

"""generate input"""
power = 5
num, times = 2**power, 100
samples = inputGen.uniform(num, times)
# x1 = np.arange(-1,1,1/16)
# x2 = np.arange(-1,1,1/16)
# samples = np.column_stack((x1,x2))
"""generate weight"""
#weight = weightGen.itentical(1/32, num)
weight = weightGen.lowPassFir(0.2, 31)

"""test effect of sc's length on the precision of filter"""
minLen = 8
maxLen = 20
CWAError = test.Test_SCLen('CWA', minLen, maxLen, 'halton', samples, weight)
HWAError = test.Test_SCLen('HWA', minLen, maxLen, 'halton', samples, weight)
MWAError = test.Test_SCLen('MWA', minLen, maxLen, 'halton', samples, weight)

"""compare result"""
xaxis = list(range(minLen, maxLen+1))
plt.plot(xaxis, CWAError, 'b-o', linewidth = 3, label='CWA')
plt.plot(xaxis, HWAError, 'g-o', linewidth = 3, label='HWA')
plt.plot(xaxis, MWAError, 'r-o', linewidth = 3, label='MWA')
for i in range(len(xaxis)):
    plt.text(xaxis[i], CWAError[i], str(round(CWAError[i], 5)))
    plt.text(xaxis[i], HWAError[i], str(round(HWAError[i], 5)))
    plt.text(xaxis[i], MWAError[i], str(round(MWAError[i], 5))) 
plt.title('Performance of different implementation of SC FIR flter change against SC length(Halton)', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()