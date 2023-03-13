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

def signedBin2Float(bin):
    """Signed binary to decimal

    Args:
        bin (1d numpy array): array of bits representing a signed float 

    Returns:
        flost: float representd by bin 
    """
    numerator = 0
    denominator = pow(2, len(bin)-1)
    for i in range(1,len(bin)):
        # bin represent a decimal
        numerator *= 2 
        numerator += bin[i]

    if (bin[0] == 1):
        numerator -= 2**(len(bin)-1)

    return numerator / denominator

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
    q = np.round(t)

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
                mask[:, j] = np.logical_and(mask[:, j], sel[:, i])    
            else:
                mask[:, j] = np.logical_and(mask[:, j], np.logical_not(sel[:, i]))
    
    input = np.logical_and(input, mask)
    return np.logical_or.reduce(input, axis=1)

input = np.array([[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],
                [0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0]])
sel = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],])
a = softMux(input, sel)
print(a.shape)
