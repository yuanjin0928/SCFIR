import numpy as np
from math import *
from datetime import datetime
from lfsr import LFSR
import tkinter as tk
from tkinter import filedialog
from shared import *

# lookup table for poly of LFSR. Start from 4 bits
LFSR_POLY = [
    [4, 1], # 9 in hex
    [5, 2], # 12
    [6, 1], # 21
    [7, 1], # 41
    [8, 4, 3, 2], # 8E
    [9, 4], # 108
    [10, 3], # 204
    [11, 2], # 402
    [12 ,6, 4, 1], # 829
    [13, 4, 3, 1], # 100D
    [14, 5, 3, 1], # 2015
    [15, 1], # 4001
    [16, 5, 3, 2], # 8016
    [17, 3], # 10004
    [18, 5, 2, 1], # 20013
    [19, 5, 2, 1], # 40013
    [20, 3], # 80004
    [21, 2], # 100002
    [22, 1], # 200001
    [23, 5], # 400010
    [24, 4, 3, 1], # 80000D
    [25, 3], # 1000004
    [26, 6, 2, 1], # 2000023
    [27, 5, 2, 1], # 4000013
    [28, 3], # 8000004
    [29, 2], # 10000002
    [30, 6, 4, 1], # 20000029
    [31, 3], # 40000004
    [32, 7, 5, 3, 2, 1] # 80000057
]

class RNS():

    #-----constructor-----#
    # parameter
    #   RNS type: current LFSR or Halton
    def __init__(self, type = 'LFSR'):
        self.type = type    # RNS type
        self.length = None  # bit length of RN
        self.poly = None    # poly for lfsr
        self.seed = None    # seed for RNS
        self.RNS  = None    # array of random number
    
    #-----set RNS'S type-----#
    # parameter
    #   type: RNS's type
    def setType(self, type):
        self.type = type
    
    #----set primitive polynomial-----#
    # parameter
    #   poly: list of coefficients  
    def setPoly(self, poly):
        self.poly = poly
    
    #----set seed-----#
    # parameter
    #   seed: seed for RNS  
    def setSeed(self, seed):
        self.seed = seed
        self.length = len(seed)
    
    #-----generate RN sequence-----# 
    # parameter
    #   num: number of RN sequences generated
    #   save: if save sequences to file
    def RNGGen(self, num, save):
        if (self.type == "LFSR"):
            return self.lfsrRNGGen(num, save)
        else:
            return self.haltonRNGGen(num, save)

    #-----LFSR RNG-----#
    # parameter
    #   num: number of RNS sequences
    #   save: if save sequences to file
    def lfsrRNGGen(self, num, save):
        lfsrRNS = LFSR(fpoly=self.poly, initstate=self.seed)
        state = lfsrRNS.state
        # state before all-0 state
        speState = np.array([0]*(self.length-1) + [1])
        outputRNS = np.empty((num, lfsrRNS.expectedPeriod+1), dtype=float)
        # add all-0 state. Total number of random number is lfsrRNS.expectedPeriod+1
        for i in range(lfsrRNS.expectedPeriod+1):
            #print(f'state: {state}')
            outputRNS[0][i] = bin2Float(state)
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
                    outputRNS[j][i] = bin2Float(rotated1)
            
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

    #def haltonRNGGen(self, num, save):
        

# r = RNS()
# for i in range(17, 21):
#     seed = []
#     for j in range(i):
#         if (j == i-1):
#             seed.append(1)
#         else:
#             seed.append(0)
#     r.setSeed(seed)
#     r.setPoly(LFSR_POLY[i-4])
#     r.RNGGen(5, True)

