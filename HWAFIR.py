import numpy as np
from shared import *

#-----implementation of Hard-wired weighted average (HWA) design-----#
class HWAFIR():

    def __init__(self):
        self.input = None
        self.weight = None

    def getInput(self, input):
        self.input = np.array(input)

    def getWeight(self, weight):
        self.weight = np.array(weight)

    def run(self, RNS, m):
        RNS = np.array(RNS)
        tap = len(self.input)
        inputSC = np.empty(RNS.shape[0], tap, dtype=bool)
        selSC = np.empty(RNS.shape[0], m, dtype=bool)

        for i in range(tap):
            inputSC[:, i] = RNS[:, 0] < self.input[i]
        for i in range(m):
            selSC[:, i] = RNS[:, 1+i] < 0.5
        
        sign = (self.weight < 0)
        productSC = np.empty(RNS.shape[0], tap, dtype=bool)
        for i in range(tap):
            productSC[:, i] = np.logical_xor(inputSC[:, i], sign)
        
        outputSC = np.empty(RNS.shape[0], dtype=bool)
        q = weightNormAndQuan(self.weight, m)
        posRef = np.zeros(tap)
        cnt = 0
        for i in range(tap):
            cnt += q[i]
            posRef[i] = cnt - 1
        
        for i in range(RNS.shape[0]):
            a = posRef - bin2Dec(selSC[i, :])
            sel = np.min(np.argwhere(a >= 0))
            outputSC[i] = productSC[i, sel]
        n1 = (outputSC == True).sum()
        n0 = (outputSC == False).sum()
        result = (n1-n0)/(n1+n0)
        return result   
        