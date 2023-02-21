import numpy as np

#-----binary to float-----#
# parameter
#   bin: array of bits representing a float   
# return
#   numerator / denominator: decimal representd by bin 
def bin2Float(bin):
    numerator = 0
    denominator = pow(2, len(bin))
    for i in range(len(bin)):
        # bin represent a decimal
        numerator *= 2  # use integer to avoid rounding error
        numerator += bin[i]
    return numerator / denominator

#-----signed binary to decimal-----#
# parameter
#   bin: array of bits representing a signed float   
# return
#   numerator / denominator: float representd by bin 
def signedBin2Float(bin):
    numerator = 0
    denominator = pow(2, len(bin)-1)
    for i in range(1,len(bin)):
        # bin represent a decimal
        numerator *= 2  # use integer to avoid rounding error
        numerator += bin[i]

    if (bin[0] == 1):
        numerator -= 2**(len(bin)-1)

    return numerator / denominator

#-----Mux tree weight normalization and quantization-----#
# parameter
#   w: weight
#   m: height of mux tree
# return
#   q: quantized and normalized weight
def weightNormAndQuan(w, m):
    a = np.array(abs(w))
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

#-----compute output sequence of a mux with given number of inputs-----#
# parameter
#   input: input to the softMux. size: n*2^sel
#   sel  : selection signal. size: n*sel
# return
#   np.logical_or.reduce(input, axis=1): output of softMux. size: n     
def softMux(input, sel):
    input = np.array(input)
    sel = np.array(sel)
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

# input = np.array([[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],
#                 [0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0],[0,1,0,0,0,1,0,0]])
# sel = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],])
# print(softMux(input, sel))

