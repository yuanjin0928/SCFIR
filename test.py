import os
import math
import numpy as np
import FIRRun

Filter = {
    'CVT': {'func': FIRRun.CVTRun, 'rns': 2},
    'HWA': {'func': FIRRun.HWARun, 'rns': 1},
    'MWA': {'func': FIRRun.MWARun, 'rns': 1},
    'OLMUX': {'func': FIRRun.OLMUXRun, 'rns': 0}
}

def Test_SCLen(arch, minLen, maxLen, samples, weight):
    """test effect of sc's length on the precision of filter"""
    CWD = os.getcwd()
    samplesPower = int(math.log2(len(weight)))
    numOfExprm = samples.shape[1]

    """calculate reference"""
    Ref = np.dot(np.transpose(samples), weight)
    error = np.empty(maxLen-minLen+1)
    for rnsLen in range(minLen, maxLen+1):
        """select rns"""
        if arch != 'OLMUX':
            rngFolder = os.path.join(CWD, 'rng/lfsr', '{}'.format(rnsLen))
            rngFiles = sorted(os.listdir(rngFolder))
            rns = np.empty((2**rnsLen, samplesPower+Filter[arch]['rns']))
            for i in range(rns.shape[1]):
                rns[:, i] = np.load(os.path.join(rngFolder,rngFiles[i]))
        else:
            pass

        """filter run"""
        result, calib = Filter[arch]['func'](samples, weight, rns) 
        error[rnsLen-minLen] = (np.sum(np.power(Ref-calib, 2))/numOfExprm)**(1/2)
    
    return error

# MWA

# OLMUX
