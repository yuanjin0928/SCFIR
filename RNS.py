import os
import numpy as np
from math import *
from datetime import datetime
from lfsr import LFSR
import shared

CWD = os.getcwd()

def lfsrRNG(size, num):
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
        rotated0 = np.roll(np.array(state), floor(self.length/2))
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
        
    if (save):
        dateTimeObj = datetime.now()
        timestamp = str(dateTimeObj.year) + str(dateTimeObj.month) + str(dateTimeObj.day) + str(dateTimeObj.hour) + str(dateTimeObj.minute)
        fileName = f'lfsr_L{self.length}_N{num}_{timestamp}.txt'
        f = open(fileName, mode='w')
        print(f'seed: {self.seed}', file=f)
        print(f'poly: {self.poly}', file=f)
        np.savetxt(f, outputRNS.T)
        f.close()   

    self.RNS = outputRNS   

def loadRNS(self):
    root = tk.Tk()
    root.withdraw()
    filePath = filedialog.askopenfilename()
    self.RNS = np.loadtxt(filePath, skiprows=2).T

# for i in range(9, 21):
#     lfsrRNG(i,8)
