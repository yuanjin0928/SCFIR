import numpy as np

#-----convert halton number to decmial value-----#
# parameter
#   state: halton number
#   base: base of halton sequence
# return
#   decimal value
def halton2dec(state, base):
    state = np.flip(np.array(state))
    numerator = 0
    denominator = base**len(state)
    for i in range(len(state)):
        numerator *= base  # use integer to avoid rounding error
        numerator += state[i]
    return numerator / denominator

class halton():

    #-----constructor-----#
    # parameter
    #   base: base of halton sequence
    def __init__(self, base):
        self.base = base

    #-----get current state-----#
    def getState(self):
        return self.state

    #-----compute next state
    def next(self):
        carryout = 0
        for i in range(len(self.state)-1, -1, -1):
            if (i == len(self.state)-1):
                if (self.state[i] == self.base-1):
                    self.state[i] = 0
                    carryout = 1
                else:
                    self.state[i] += 1
            else:
                if (carryout and self.state[i] == self.base-1):
                    self.state[i] = 0
                elif (carryout):
                    self.state[i] += 1
                    carryout = 0
        if (np.sum(self.state) == 0):
            self.state[-1] = 1   
        return self.state

    #-----set new seed
    # parameter
    #   seed: seed
    def setSeed(self, seed):
        self.seed = np.array(seed)
        self.state = np.array(seed)
    
    # def halton(b):
    #     n, d = 0, 1   # n: numerator d: denominator
    #     while True:
    #         x = d - n # x: difference between n and d
    #         if x == 1:    # have reached the last value of last level
    #             n = 1
    #             d *= b
    #         else:
    #             y = d // b    # how many parts last level have
    #             while x <= y: # if difference between n and d is smaller than y, which means value lies in the last part has been computed
    #                 y //= b   # overflow. b-x+1
    #             n = (b + 1) * y - x # cycle = y, last element d-x, next element d-x+y, d=y*b, next element y+b*y-x = (b+1)*y-x                                    
    #         yield n / d

#-----try correlation-----#
w = 4  # sc resolution
base = 3 # halton base
seedLen = int(np.ceil(np.log(2**w+1)/np.log(3)))
seed = []
for i in range(seedLen):
    if (i == seedLen-1):
        seed.append(1)
    else:
        seed.append(0)
for k in range(seedLen): 
    sccTotal = 0
    for i in range(2**w):
        for j in range(2**w):
            a = i / 2**w
            b = j / 2**w
            h = halton(base)
            h.setSeed(seed)
            aSC = np.empty(2**w, dtype=bool)
            bSC = np.empty(2**w, dtype=bool)
            for m in range(2**w):
                state = h.getState()
                stateRotated = np.roll(state, k)
                if (halton2dec(state, base) < a):
                    aSC[m] = True
                else:
                    aSC[m] = False  

                if(halton2dec(stateRotated, base) < b):
                    bSC[m] = True
                else:
                    bSC[m] = False 
                
                h.next()

            scProduct = np.sum(np.bitwise_and(aSC, bSC)) / 2**w
            delta = scProduct - a*b
            if (delta > 0):
                scc = delta / (np.min([a,b])-a*b)
            elif(delta < 0):
                scc = np.abs(delta / (a*b-np.max([a+b-1,0])))    
            else:
                scc = 0

            sccTotal += scc     

    sccAve = sccTotal / 2**(w+w)   
    print(sccAve)
            


