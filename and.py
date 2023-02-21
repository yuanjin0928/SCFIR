import numpy as np
import matplotlib.pyplot as plt
from RNS import *

#-----performance simulation of SC multiplication implemented by AND gate-----#

#-----rmse changes against input length with two input sharing same lfsr-----#
'''
    Note: two other experiments have also been performed.
    1: The input length and the lfsr length are both fixed, and I iterate all the seeds. The rmse of each expriment with unique seed are same
    2: The input length is fixed, and I increase the lfsr length by one each time. The rmse of each experiment with different lfsr are same
'''
def lfsrVSlfsr_1():
    iteRange = list(range(4,11))
    r = RNS()
    rmse = np.empty(len(iteRange))
    cnt = 0
    for i in iteRange:
        r.setPoly(LFSR_POLY[i-4])
        err = 0 
        seed = []
        for j in range(i):
            if (j == i-1):
                seed.append(1)
            else:
                seed.append(0)  
        r.setSeed(seed)
        r.RNGGen(1, False)
        rng = np.array(r.RNS)

        input0 = np.empty(2**i, dtype=bool)
        input1 = np.empty(2**i, dtype=bool)
        scResult = np.empty(2**i,dtype=bool)
        for j in range(2**i):
            for k in range(2**i):
                a = j / 2**i
                b = k / 2**i 
                input0 = rng < a
                input1 = rng < b
                scResult = np.bitwise_and(input0, input1)
                scProduct = np.sum(scResult) / 2**i
                binProduct = a * b
                errSquare = pow(binProduct-scProduct, 2)
                err += errSquare
        
        errAve = pow(err / 2**(i*2), 1/2)
        rmse[cnt] = errAve
        cnt += 1
    
    plt.plot(rmse)
    plt.show()

#-----rmse changes against different input combinations-----#
'''
    Note: the maximal rmse is achieved when two inputs are both one half 
'''
def lfsrVSlfsr_2(inputLen):
    r = RNS()
    rmse = np.empty(2**(inputLen*2))
    r.setPoly(LFSR_POLY[inputLen-4])
    seed = []
    for j in range(inputLen):
        if (j == inputLen-1):
            seed.append(1)
        else:
            seed.append(0)  
    r.setSeed(seed)
    r.RNGGen(1, False)
    rng = np.array(r.RNS)

    input0 = np.empty(2**inputLen, dtype=bool)
    input1 = np.empty(2**inputLen, dtype=bool)
    scResult = np.empty(2**inputLen,dtype=bool)

    cnt = 0
    for j in range(2**inputLen):
        for k in range(2**inputLen):
            a = j / 2**inputLen
            b = k / 2**inputLen 
            input0 = rng < a
            input1 = rng < b
            scResult = np.bitwise_and(input0, input1)
            scProduct = np.sum(scResult) / 2**inputLen
            binProduct = a * b
            err = abs(binProduct-scProduct) / binProduct
            rmse[cnt] = err
            cnt += 1
    
    plt.plot(rmse,'-.o')
    plt.show()

#-----two lfsr to generate rn with different seeds-----#
def lfsrVSlfsr_3(inputLen, lfsrLen):
    r1 = RNS()
    r2 = RNS()
    r1.setPoly(LFSR_POLY[lfsrLen-4])
    r2.setPoly(LFSR_POLY[lfsrLen-4])
    rmse = np.empty((2**lfsrLen-1)*(2**lfsrLen-1))

    cnt = 0
    for i in range(1,2**lfsrLen):
        for j in range(1, 2**lfsrLen):
            err = 0 
            seed1 = []
            seed2 = []

            for k in range(lfsrLen):
                seed1.insert(0, (i>>k)&1)
                seed2.insert(0, (j>>k)&1)
                
            r1.setSeed(seed1)
            r2.setSeed(seed2)
            r1.RNGGen(1, False)
            r2.RNGGen(1, False)
            rng1 = np.array(r1.RNS)
            rng2 = np.array(r2.RNS)

            input1 = np.empty(2**lfsrLen, dtype=bool)
            input2 = np.empty(2**lfsrLen, dtype=bool)  
            for m in range(2**inputLen):
                for n in range(2**inputLen):
                    a = m / 2**inputLen
                    b = n / 2**inputLen 
                    input1 = rng1 < a
                    input2 = rng2 < b
                    scResult = np.bitwise_and(input1, input2)
                    scProduct = np.sum(scResult) / 2**lfsrLen
                    binProduct = a * b
                    errSquare = pow(binProduct-scProduct, 2)
                    err += errSquare

            errAve = pow(err / 2**(inputLen*2), 1/2)
            rmse[cnt] = errAve
            cnt += 1
    
    plt.plot(rmse, '-.o')
    plt.show()


def lfsrVSlfsr_4(inputLen, lfsrLen):
    r = RNS()
    r.setPoly(LFSR_POLY[lfsrLen-4])
    seed = []
    for j in range(lfsrLen):
        if (j == lfsrLen-1):
            seed.append(1)
        else:
            seed.append(0)  
    r.setSeed(seed)
    r.RNGGen(lfsrLen, False)
    rng = np.array(r.RNS)
    rmse = np.empty(int(lfsrLen*(lfsrLen-1)/2))

    cnt = 0
    for i in range(lfsrLen-1):
        for j in range(i+1, lfsrLen):
            err = 0
            for m in range(2**inputLen):
                for n in range(2**inputLen):
                    a = m / 2**inputLen
                    b = n / 2**inputLen 
                    input1 = rng[i] < a
                    input2 = rng[j] < b
                    scResult = np.bitwise_and(input1, input2)
                    scProduct = np.sum(scResult) / 2**lfsrLen
                    binProduct = a * b
                    errSquare = pow(binProduct-scProduct, 2)
                    err += errSquare
            errAve = pow(err / 2**(inputLen*2), 1/2)
            rmse[cnt] = errAve
            cnt += 1   

    plt.plot(rmse, '-.o')
    plt.show()         

#lfsrVSlfsr_3(4 ,8)
    
                    


#def lfsrVshalton():

#def haltonVshalton():
