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
"""generate weight"""
#weight = weightGen.itentical(1/32, num)
weight = weightGen.lowPassFir(0.2, 31)

"""test effect of sc's length on the precision of filter"""
minLen = 8
maxLen = 20

# """lfsr as RNS"""
# CWALfsrError = test.Test_SCLen('CWA', minLen, maxLen, 'lfsr', samples, weight)
# HWALfsrError = test.Test_SCLen('HWA', minLen, maxLen, 'lfsr', samples, weight)
# MWALfsrError = test.Test_SCLen('MWA', minLen, maxLen, 'lfsr', samples, weight)
# OLMUXLfsrError = test.Test_SCLen('OLMUX', minLen, maxLen, 'lfsr', samples, weight)

# """halton as RNS"""
# CWAHaltonError = test.Test_SCLen('CWA', minLen, maxLen, 'halton', samples, weight)
# HWAHaltonError = test.Test_SCLen('HWA', minLen, maxLen, 'halton', samples, weight)
# MWAHaltonError = test.Test_SCLen('MWA', minLen, maxLen, 'halton', samples, weight)
# OLMUXHaltonError = test.Test_SCLen('OLMUX', minLen, maxLen, 'halton', samples, weight)

"""lfsr as RNS"""
CWALfsrError = test.Test_DeterministicSel('CWA', minLen, maxLen, 'lfsr', samples, weight)
HWALfsrError = test.Test_DeterministicSel('HWA', minLen, maxLen, 'lfsr', samples, weight)

"""halton as RNS"""
CWAHaltonError = test.Test_DeterministicSel('CWA', minLen, maxLen, 'halton', samples, weight)
HWAHaltonError = test.Test_DeterministicSel('HWA', minLen, maxLen, 'halton', samples, weight)

"""plot result"""
fig, (plot1, plot2) = plt.subplots(2,1, figsize=(8, 6))
xaxis = list(range(minLen, maxLen+1))

plot1.plot(xaxis, CWALfsrError, 'b-o', linewidth = 3, label='CWA')
plot1.plot(xaxis, HWALfsrError, 'g-o', linewidth = 3, label='HWA')
# plot1.plot(xaxis, MWALfsrError, 'r-o', linewidth = 3, label='MWA')
# plot1.plot(xaxis, OLMUXLfsrError, 'k-o', linewidth=3, label='OLMUX')
for i in range(len(xaxis)):
    plot1.text(xaxis[i], CWALfsrError[i], str(round(CWALfsrError[i], 5)))
    plot1.text(xaxis[i], HWALfsrError[i], str(round(HWALfsrError[i], 5)))
    # plot1.text(xaxis[i], MWALfsrError[i], str(round(MWALfsrError[i], 5))) 
    # plot1.text(xaxis[i], OLMUXLfsrError[i], str(round(OLMUXLfsrError[i], 5)))
plot1.set_title('RMSE of different implementation of SC FIR flter changes against SC length with deterministic selection signal(lfsr)', fontsize=20)
plot1.legend(fontsize=14)
plot1.tick_params(axis='both', which='major', labelsize=14)

plot2.plot(xaxis, CWAHaltonError, 'b-o', linewidth = 3, label='CWA')
plot2.plot(xaxis, HWAHaltonError, 'g-o', linewidth = 3, label='HWA')
# plot2.plot(xaxis, MWAHaltonError, 'r-o', linewidth = 3, label='MWA')
# plot2.plot(xaxis, OLMUXHaltonError, 'k-o', linewidth = 3, label='OLMUX')
for i in range(len(xaxis)):
    plot2.text(xaxis[i], CWAHaltonError[i], str(round(CWAHaltonError[i], 5)))
    plot2.text(xaxis[i], HWAHaltonError[i], str(round(HWAHaltonError[i], 5)))
    # plot2.text(xaxis[i], MWAHaltonError[i], str(round(MWAHaltonError[i], 5))) 
    # plot2.text(xaxis[i], OLMUXHaltonError[i], str(round(OLMUXHaltonError[i], 5)))
plot2.set_title('RMSE of different implementation of SC FIR flter changes against SC length with deterministic selection signal(halton)', fontsize=20)
plot2.legend(fontsize=14)
plot2.tick_params(axis='both', which='major', labelsize=14)

plt.show()