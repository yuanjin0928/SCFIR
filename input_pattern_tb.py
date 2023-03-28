import os
import numpy as np
import matplotlib.pyplot as plt
import inputGen
import weightGen
import test

Input_Gen_Func = {
    'uniform': inputGen.uniform,
    'normal' : inputGen.nomal
}

"""specify pattern"""
pattern = [
    {'type': 'uniform', 'para': [-1,1]},
    {'type':'uniform', 'para': [-3/4, 3/4]},
    {'type':'uniform', 'para': [-1/2, 1/2]},
    {'type':'uniform', 'para': [-1/4, 1/4]},
    {'type':'normal',  'para': [0, 1/4]},
    {'type':'normal',  'para': [0, 1/2]},
    {'type':'normal',  'para': [0, 3/4]},
    {'type':'normal',  'para': [0, 1]}
]

"""generate input"""
power = 7
num, times = 2**power, 100

samples = np.empty((num, times, len(pattern)))
for i in range(len(pattern)):
    samples[:, :, i] = Input_Gen_Func[pattern[i]['type']](pattern[i]['para'], num, times)
"""generate weight"""
weight = weightGen.lowPassFir(0.2, num-1)

"""run filter"""
CWAError = test.Test_Input_Pattern('CWA', samples, weight)
HWAError = test.Test_Input_Pattern('HWA', samples, weight)
MWAError = test.Test_Input_Pattern('MWA', samples, weight)
OLMUXError = test.Test_Input_Pattern('OLMUX', samples, weight)

# CWAError = list(range(len(pattern)))
# HWAError = list(range(len(pattern)))
# MWAError = list(range(len(pattern)))
# OLMUXError = list(range(len(pattern)))


"""plot result"""
xaxis = list(range(len(pattern)))

plt.plot(xaxis, CWAError, 'b-o', linewidth = 3, label='CWA')
plt.plot(xaxis, HWAError, 'g-o', linewidth = 3, label='HWA')
plt.plot(xaxis, MWAError, 'r-o', linewidth = 3, label='MWA')
plt.plot(xaxis, OLMUXError, 'k-o', linewidth=3, label='OLMUX')
for i in range(len(xaxis)):
    plt.text(xaxis[i], CWAError[i], str(round(CWAError[i], 5)))
    plt.text(xaxis[i], HWAError[i], str(round(HWAError[i], 5)))
    plt.text(xaxis[i], MWAError[i], str(round(MWAError[i], 5))) 
    plt.text(xaxis[i], OLMUXError[i], str(round(OLMUXError[i], 5)))

for i in range(len(pattern)):
    text = '{}: {}, {}'.format(pattern[i]['type'], pattern[i]['para'][0], pattern[i]['para'][1])
    plt.text(0.95, 0.95-0.05*i, text, fontsize = 16, ha='right', va='top', transform=plt.gca().transAxes)

plt.title('RMSE of different SC-based FIR filter with input of various statistical distributions', fontsize=20)
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.show()