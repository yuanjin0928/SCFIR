import numpy as np
from shared import *

#-----implementation of convential stochastic FIR filter-----#
class cvtFIR():

    def __init__(self):
        self.weight = None
        self.input = None

    def getInput(self, input):
        self.input = np.array(input)


    def getWeight(self, weight):
        self.weight = np.array(weight)

    def run(self, RNS):
        RNS = np.array(RNS)
        tap = len(self.input)
        selNum = np.log2(tap)
        inputSC = np.empty(RNS.shape[0], tap, dtype=bool)
        weightSC = np.empty(RNS.shape[0], tap, dtype=bool)
        selSC = np.empty(RNS.shape[0], selNum, dtype=bool)

        for i in range(tap):
            inputSC[:, i] = RNS[:, 0] < self.input[i]
            weightSC[:, i] = RNS[:, 1] < self.weight[i]
        for i in range(selNum):
            selSC[:, i] = RNS[:, 2+i] < 0.5
        
        productSC = np.logical_not(np.logical_xor(inputSC, weightSC))
        outputSC = np.empty(RNS.shape[0], dtype=bool)
        for i in range(RNS.shape[0]):
            outputSC[i] = productSC[i, bin2Float(selSC[i, :])]
        
        n1 = (outputSC == True).sum()
        n0 = (outputSC == False).sum()
        result = (n1-n0)/(n1+n0)
        return result
        
        



    

