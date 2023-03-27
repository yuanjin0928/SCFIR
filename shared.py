import numpy as np

"""
    Useful public functions
"""

def bin2Float(bin):
    """Binary to float

    Args:
        bin (1d numpy array): array of bits representing a float

    Returns:
        flost: decimal representd by bin
    """
    numerator = 0
    denominator = pow(2, len(bin))
    for i in range(len(bin)):
        # bin represent a decimal
        numerator *= 2 
        numerator += bin[i]
    return numerator / denominator

def int2binArray(dec, size):
    return np.flip(np.array([(dec >> i) & 1 for i in range(size)]))

def weightNormAndQuan(w, m):
    """Mux tree weight normalization and quantization

    Args:
        w (1d numpy array): weight
        m (int): height of mux tree

    Returns:
        1d numpy array: quantized and normalized weight
    """
    a = np.abs(w)
    sum = np.sum(a)
    t = a / sum * 2**m
    q = np.round(t).astype(int)

    while (np.sum(q) > 2**m):
        i = np.argmax(q-t)
        q[i] = q[i] - 1

    while (np.sum(q) < 2**m):
        i = np.argmax(t-q)
        q[i] = q[i] + 1

    return q


def softMux(input, sel):
    """Compute output sequence of a mux with given number of inputs

    Args:
        input (2d numpy array): input to the softMux. size: SNLen*2^sel
        sel (2d numpy array): selection signal. size: SNLen*sel

    Returns:
        1d numpy array: output of mux
    """
    mask = np.ones(input.shape, dtype=bool)

    for i in range(sel.shape[1]): 
        shift = sel.shape[1]-1-i
        for j in range(mask.shape[1]):
            quotient = np.right_shift(j,shift)
            if (quotient % 2 == 1):
                # consider each column of input array as one input to the mux, and perform selection
                mask[:, j] = np.logical_and(mask[:, j], sel[:, i])    
            else:
                mask[:, j] = np.logical_and(mask[:, j], np.logical_not(sel[:, i]))
    
    input = np.logical_and(input, mask)
    return np.logical_or.reduce(input, axis=1)

def correlation(input0, input1):
    p0 = np.sum(input0) / input0.shape[0]
    p1 = np.sum(input1) / input1.shape[0]
    delta = np.sum(np.logical_and(input0, input1))/input0.shape[0] - p0*p1
    if delta == 0:
        return 0
    elif delta > 0:
        return delta/(min(p0, p1) - p0*p1)
    else:
        return delta/(p0*p1 - max(p0+p1-1,0))
