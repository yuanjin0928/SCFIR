# import wfdb
import numpy as np
import matplotlib.pyplot as plt
from RNS import RNS
from fir import fir

if __name__ == "__main__":

    #-----get data from dataset-----#
    channels = [0]  
    sampleFrom = 40
    sampleLen = 100
    # load a record using the 'rdrecord' function
    sampleFolder = 'mit-bih-arrhythmia-database-1.0.0/'
    fileName = '101'
    record = wfdb.rdrecord(sampleFolder+fileName, sampleFrom, sampleFrom+sampleLen, channels, physical=False, return_res=16)

    sampleSoft = np.empty(sampleLen)
    sampleSC = np.empty(sampleLen)
    for i in range(sampleLen):
        sampleSoft[i] = (record.d_signal[i][0] - 1024)
        sampleSC[i] = sampleSoft[i] / 2**10


    #-----software-based fir-----#
    taps = 32
    cutoffFreq = 30
    window = 'hann'
    f = fir()
    h = f.filterDesign(taps, cutoffFreq, record.fs, window)
    
    softResult = f.filter(sampleSoft, h)

    #-----sc-based fir-----#

    # muxTree
    m = muxTree(h)
    m.buildTree()
    r = RNS()
    r.loadRNS()
    #r.RNGGen(len(m.Tree), True)
    muxTreeResult = m.run(sampleSC, r.RNS)
    muxTreeResult = muxTreeResult * 2**10

    #-----sbow result-----#
    delay = int(((taps-1)/2))
    pointNum = sampleLen - delay
    t = np.array(range(pointNum))
    plt.plot(t, sampleSoft[0:pointNum],'r-.o', t, softResult[delay:sampleLen], 'g-.o', t, muxTreeResult[delay:sampleLen], 'b-.o')
    plt.show()

    #-----compute error
    errSum = 0
    for i in range(pointNum):
        errSum = errSum + pow(softResult[delay+i]-muxTreeResult[delay+i],2)  
    
    errSum = pow(errSum/pointNum, 1/2)
    print(errSum)

