import os
import numpy as np
import matplotlib.pyplot as plt
import inputGen
import weightGen
import test

"""generate input"""
power = 5
num, times = 2**power, 100
para = [-1,1]
samples = inputGen.uniform(para, num, times)
"""generate weight"""
weight = weightGen.lowPassFir(0.2, 31)

"""test effect of sc's length on the precision of filter"""
minLen = 8
maxLen = 20

"""run filter"""
CWAError = test.Test_Input_Lfsr_Sel_Halton('CWA', minLen, maxLen, samples, weight)
HWAError = test.Test_Input_Lfsr_Sel_Halton('HWA', minLen, maxLen, samples, weight)
MWAError = test.Test_Input_Lfsr_Sel_Halton('MWA', minLen, maxLen, samples, weight)
OLMUXError = test.Test_Input_Lfsr_Sel_Halton('OLMUX', minLen, maxLen, samples, weight)

# CWAError = list(range(minLen, maxLen+1))
# HWAError = list(range(minLen, maxLen+1))
# MWAError = list(range(minLen, maxLen+1))
# OLMUXError = list(range(minLen, maxLen+1))

"""plot result"""
xaxis = list(range(minLen, maxLen+1))

plt.plot(xaxis, CWAError, 'b-o', linewidth = 3, label='CWA')
plt.plot(xaxis, HWAError, 'g-o', linewidth = 3, label='HWA')
plt.plot(xaxis, MWAError, 'r-o', linewidth = 3, label='MWA')
plt.plot(xaxis, OLMUXError, 'k-o', linewidth=3, label='OLMUX')
for i in range(len(xaxis)):
    plt.text(xaxis[i], CWAError[i], str(round(CWAError[i], 5)))
    plt.text(xaxis[i], HWAError[i], str(round(HWAError[i], 5)))
    plt.text(xaxis[i], MWAError[i], str(round(MWAError[i], 5))) 
    plt.text(xaxis[i], OLMUXError[i], str(round(OLMUXError[i], 5)))
plt.title('RMSE of different SC-based FIR filter with Lfsr as input and Halton as Selection', fontsize=20)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()