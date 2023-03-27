import os
import math
import numpy as np
from lfsr import LFSR
import shared

def lfsrRNG(size, num):
    CWD = os.getcwd()
    lfsrPolyFile = os.path.join(CWD, 'rng/lfsr_poly', '{}'.format(size) + '.DAT')
    with open(lfsrPolyFile) as f:
        # discard description
        next(f)
        next(f)
        next(f)
        next(f)
        cnt = 0
        criticalState = np.array([0]*(size-1) + [1])
        for line in f:
            cnt += 1
            line = line.rstrip('\r\n')
            lfsrRNFile = os.path.join(CWD, 'rng/lfsr', '{}'.format(size), line)
            if not os.path.isfile(lfsrRNFile+'.npy'):
                RNArray = np.empty(2**size)
                polyBin = format(int(line, 16), 'b')
                polyList = [i for i, digit in enumerate(reversed(polyBin), 1) if digit == '1']
                polyList.reverse()
                lfsrRNS = LFSR(fpoly=polyList)
                state = lfsrRNS.state
                for i in range(lfsrRNS.expectedPeriod + 1):
                    RNArray[i] = shared.bin2Float(state)
                    if np.array_equal(state, criticalState):
                        state = [0] * size
                    else:
                        state = lfsrRNS.next()
                np.save(lfsrRNFile, RNArray)
            if cnt == num:
                break

def haltonRNG(num, length):
    """Generate halton sequences
    From Florian's halton.py for SC based decryption
    Args:
        num (int): num of sequences 
        length (int): number of rn in each sequence
    
    Returns:
        array: halton sequence
    """
    CWD = os.getcwd()

    big_number = 10
    while 'Not enough primes':
        base = primes_from_2_to(big_number)[:num]
        if len(base) == num:
            break
        big_number += 1000

    haltonDir = os.path.join(CWD, 'rng/halton')
    folders = os.listdir(haltonDir)
    folders = sorted([int(element) for element in folders])
    if length > max(folders):
        for primes in base:
            haltonFileName = os.path.join(CWD, 'halton', '{}'.format(length), '{}'.format(primes))
            np.save(haltonFileName, van_der_corput(length, primes))
    elif length < min(folders):
        haltonSubDir = os.path.join(haltonDir, '{}'.format(min(folders)))
        for primes in base:
            haltonFileName = os.path.join(haltonSubDir, '{}'.format(primes))
            if not os.path.isfile(haltonFileName + '.npy'):
                np.save(haltonFileName, van_der_corput(min(folders), primes))    
    else:
        haltonSubDir = os.path.join(haltonDir, '{}'.format(length))
        for primes in base:
            haltonFileName = os.path.join(haltonSubDir, '{}'.format(primes))
            if not os.path.isfile(haltonFileName + '.npy'):
                np.save(haltonFileName, van_der_corput(min(folders), primes))    

def primes_from_2_to(n):
    """Prime number from 2 to n
    From stackoverflow <https://stackoverflow.com/questions/2068372>

    Args:
        n (int): sup bound with n >= 6

    Returns:
        list: primes in 2 <= p < n 
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1,int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    
    return np.r_[2,3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

def van_der_corput(length, base):
    sequence = []
    cnt = 0
    for i in range(2**length+1):
        n_th_number, denom = 0., 1.
        j = i
        while j > 0:
            j, remainder = divmod(j, base)
            denom *= base
            n_th_number += remainder / denom
        
        if i > 0:
            sequence.append(n_th_number)
            cnt += 1
    
    return np.array(sequence)


def lfsrRNGGen(self, num, save):
    lfsrRNS = LFSR(fpoly=self.poly, initstate=self.seed)
    state = lfsrRNS.state
    # state before all-0 state
    speState = np.array([0]*(self.length-1) + [1])
    outputRNS = np.empty((num, lfsrRNS.expectedPeriod+1), dtype=float)
    # add all-0 state. Total number of random number is lfsrRNS.expectedPeriod+1
    for i in range(lfsrRNS.expectedPeriod+1):
        #print(f'state: {state}')
        outputRNS[0][i] = shared.bin2Float(state)
        rotated0 = np.roll(np.array(state), math.floor(self.length/2))
        #print(f'rotated0: {rotated0}')
        # add all-0 state
        if (np.array_equal(state, speState)):
            state = np.array([0]*(self.length))
        else:
            lfsrRNS.next()
            state = lfsrRNS.state
        cnt = 0
        if (num > 1):
            for j in range(1,num):
                if (j%2 == 0):
                    cnt = cnt + 1
                    rotated1 = np.roll(rotated0, cnt)
                else:
                    rotated1 = np.roll(rotated0, -cnt)
                #print(f'rotated1: {rotated1}')
                outputRNS[j][i] = shared.bin2Float(rotated1)

    self.RNS = outputRNS 

def deterministicSel(rnsLen, samplesPower):
    rns = np.empty((2**rnsLen, samplesPower))
    sliceLen = rnsLen - samplesPower
    for i in range(2**samplesPower):
        binArray = shared.int2binArray(i, samplesPower)
        selArray = np.array([(x+0.5) if (x==0) else 0 for x in binArray])
        rns[i*2**sliceLen:(i+1)*2**sliceLen, :] = selArray
    
    return rns