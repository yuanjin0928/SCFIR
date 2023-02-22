
def cvtRun(self, RNS):
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

def HWARun(self, RNS, m):
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