import numpy as np
from shared import *

def cvtRun(input, weight, rns):
    """Conventional design of FIR filter

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        float: result of the inner product
    """
    input = np.transpose(input)
    weight = np.transpose(weight)

    # Convert input and weight to bipolar represented SN
    inputSC = rns[:, 0] < (input+1)/2
    weightSC = rns[:, 1] < (weight+1)/2
    # Compute unipolar represented selection signal
    selSC = rns[:, 2:] < 0.5
    
    # Use xnor to perform multiplication of input and weight 
    productSC = np.logical_not(np.logical_xor(inputSC, weightSC))
    # Perform scaled addition
    outputSC = softMux(productSC, selSC)
    
    n1 = (outputSC == True).sum()
    n0 = (outputSC == False).sum()
    result = (n1-n0)/(n1+n0)
    return result


def HWARun(input, weight, height, rns):
    """Hard-wired weighted average design of FIR filter

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        height (int): height of MUX
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        float: result of the inner product
    """
    weight = np.transpose(weight)
    # Normalized and quantized weight
    q = weightNormAndQuan(weight, height)

    # Replicate input according to individual weight
    inputExt = np.transpose(np.empty(2**height))
    cnt = 0
    for i in range(len(q)):
        num = q[i]
        for j in range(num):
            inputExt[cnt] = input[i]
            cnt = cnt + 1

    # Convert input to bipolar represented SN
    inputSC = rns[:, 0] < (inputExt+1)/2
    # Compute unipolar represented selection signal
    selSC = rns[:, 1:] < 0.5
    sign = weight < 0
    
    productSC = np.logical_xor(inputSC, sign)
    outputSC = softMux(productSC, selSC)
    
    n1 = (outputSC == True).sum()
    n0 = (outputSC == False).sum()
    result = (n1-n0)/(n1+n0)
    return result   