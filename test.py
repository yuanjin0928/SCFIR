import os
import math
import numpy as np
import bisect
import FIRRun
import RNS

Filter = {
    'CWA': {'func': FIRRun.CWARun, 'rns': 2},
    'HWA': {'func': FIRRun.HWARun, 'rns': 1},
    'MWA': {'func': FIRRun.MWARun, 'rns': 1},
    'OLMUX': {'func': FIRRun.OLMUXRun, 'rns': 1}
}

def Test_SCLen(arch, minLen, maxLen, rnType, samples, weight):
    """test effect of sc's length on the precision of filter"""
    CWD = os.getcwd()
    samplesPower = int(math.log2(len(weight)))
    numOfExprm = samples.shape[1]

    """calculate reference"""
    Ref = np.dot(np.transpose(samples), weight)
    error = np.empty(maxLen-minLen+1)
    if rnType == 'lfsr':
        for rnsLen in range(minLen, maxLen+1):
            """select rns"""
            rngFolder = os.path.join(CWD, 'rng/lfsr', '{}'.format(rnsLen))
            rngFiles = sorted(os.listdir(rngFolder))
            rns = np.empty((2**rnsLen, samplesPower+Filter[arch]['rns']))
            for i in range(rns.shape[1]):
                rns[:, i] = np.load(os.path.join(rngFolder,rngFiles[i]))

            """filter run"""
            result, calib = Filter[arch]['func'](samples, weight, rns) 
            error[rnsLen-minLen] = (np.sum(np.power(Ref-calib, 2))/numOfExprm)**(1/2)
    elif rnType == 'halton':
        haltonDir = os.path.join(CWD, 'rng/halton')   
        folders = os.listdir(haltonDir)
        folders = sorted([int(element) for element in folders])
        index = bisect.bisect_left(folders, maxLen)
        haltonSubDir = os.path.join(haltonDir, '{}'.format(folders[index]))
        numRns = samplesPower + Filter[arch]['rns']
        rnsSource = np.empty((2**folders[index], numRns))
        files = os.listdir(haltonSubDir)
        files = sorted([int(element.rstrip('.npy')) for element in files])
        for i in range(numRns):
            fileName = os.path.join(haltonSubDir, '{}.npy'.format(files[i]))
            rnsSource[:, i] = np.load(fileName)
        
        for rnsLen in range(minLen, maxLen+1):
            rns = rnsSource[:2**rnsLen, :]
            result, calib = Filter[arch]['func'](samples, weight, rns) 
            error[rnsLen-minLen] = (np.sum(np.power(Ref-calib, 2))/numOfExprm)**(1/2)
    
    return error

def Test_DeterministicSel(arch, minLen, maxLen, rnType, samples, weight):
    CWD = os.getcwd()
    samplesPower = int(math.log2(len(weight)))
    numOfExprm = samples.shape[1]

    """calculate reference"""
    Ref = np.dot(np.transpose(samples), weight)
    error = np.empty(maxLen-minLen+1)
    if rnType == 'lfsr':
        for rnsLen in range(minLen, maxLen+1):
            """select rns"""
            rngFolder = os.path.join(CWD, 'rng/lfsr', '{}'.format(rnsLen))
            rngFiles = sorted(os.listdir(rngFolder))
            rns = np.empty((2**rnsLen, samplesPower+Filter[arch]['rns']))
            for i in range(Filter[arch]['rns']):
                rns[:, i] = np.load(os.path.join(rngFolder,rngFiles[i]))

            rns[:, Filter[arch]['rns']:] = RNS.deterministicSel(rnsLen, samplesPower)
            """filter run"""
            result, calib = Filter[arch]['func'](samples, weight, rns) 
            error[rnsLen-minLen] = (np.sum(np.power(Ref-calib, 2))/numOfExprm)**(1/2)
    elif rnType == 'halton':
        haltonDir = os.path.join(CWD, 'rng/halton')   
        folders = os.listdir(haltonDir)
        folders = sorted([int(element) for element in folders])
        index = bisect.bisect_left(folders, maxLen)
        haltonSubDir = os.path.join(haltonDir, '{}'.format(folders[index]))
        numRns = samplesPower + Filter[arch]['rns']
        rnsSource = np.empty((2**folders[index], Filter[arch]['rns']))
        files = os.listdir(haltonSubDir)
        files = sorted([int(element.rstrip('.npy')) for element in files])
        for i in range(Filter[arch]['rns']):
            fileName = os.path.join(haltonSubDir, '{}.npy'.format(files[i]))
            rnsSource[:, i] = np.load(fileName)
        
        for rnsLen in range(minLen, maxLen+1):
            rns = np.empty((2**rnsLen, numRns))
            rns[:, :Filter[arch]['rns']] = rnsSource[:2**rnsLen, :]
            rns[:, Filter[arch]['rns']:] = RNS.deterministicSel(rnsLen, samplesPower)
            result, calib = Filter[arch]['func'](samples, weight, rns) 
            error[rnsLen-minLen] = (np.sum(np.power(Ref-calib, 2))/numOfExprm)**(1/2)
    
    return error
# MWA

# OLMUX
