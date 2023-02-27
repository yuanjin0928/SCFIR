import numpy as np
from math import *
from shared import *

def cvtRun(input, weight, rns):
    """Conventional design of a FIR filter

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        float: output of the FIR filter at one time point
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
    """Hard-wired weighted average design of a FIR filter

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        height (int): height of MUX
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        float: output of the FIR filter at one time point
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

def MWARun(input, weight, rns):

def OLMUXRun(input, weight, rns):
    """Lowest optimum MUX tree design of a FIR filter

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        rns (1d numpy array): used to convert input and weight to stochastic numbers
    
    Returns:
    float: output of the FIR filter at one time point
    """
    # build lowest optimum tree
    muxTree = OLMUXBuildTree(weight)
    
    result = OLMUXFunc(input, weight, muxTree, rns)
    
    return result


def CeMuxRun(input, weight, height, rns):


def OLMUXBuildTree(weight):
    """Build lowest optimum MUX tree

    Args:
        weight (list of float): list of weights of FIR filter

    Returns:
        list: architecture of OLMUX tree
    """
    taps = len(weight)
    D = ceil(log2(taps))  # height of MUX tree
    Q = [] # packages 
    depth = []  # depth of all the coefficients
    for i in range(taps):
        depth.append([weight[i],i, 0])  # Initialization 
    Q0 = [] # Q0 is a list
    for i in range(taps):
        #[H[i], i, -1, -1], first entry: value, second entry: position in the given coeffcients list, third entry: position of left child in the previous package, fourth entry: position of right child in the previous package
        Q0.append([abs(weight[i]), i, -1, -1])    
    Q0.sort()
    Q.append(Q0)    # Q is a list of list
    
    # len of Q: D
    for i in range(D-1):
        Q.append(OLMUXMerge(Q[0],OLMUXPackage(Q[i])))   # calculate all packages
    
    childNumbering = list(range(len(Q[D-1])))   # store the position of nodes which will be accessed in current loop. initial value: postion of all root nodes which are also the nodes in the last package
    # traverse in reverse order
    for i in range(D-1,-1,-1):
        nextChildNumbering = [] # position of current package's children in the previous package
        for j in range(len(childNumbering)):
            childPos = childNumbering[j]    # get position of child
            child = Q[i][childPos] 
            if(child[2] == -1): # leaf node, increase depth
                elementPos = child[1]   # get postion of this value in the initial coefficient list
                depth[elementPos][2] = depth[elementPos][2] + 1 # increase its depth
            else:   # not a leaf node, get position of its two children
                leftChildNumbering = child[2]
                rightChildNumbering = child[3]
                nextChildNumbering.append(leftChildNumbering)
                nextChildNumbering.append(rightChildNumbering)
        childNumbering = nextChildNumbering
    depth.sort(key=lambda x: x[2], reverse=True)    # sort depth in descending order to make the following tree construction easier
    
    # build MUX tree
    tree = []
    startingPos = 0 # since depth list is already sorted, recording the starting postion of each layer makes the construction efficient.
    for i in range(D, 0, -1):
        depthN = []
        for j in range(startingPos, taps):
            element = depth[j]
            if (element[2] == i):   # this element belongs to layer i, adding it to depthN
                entry = {
                    'value': element[0],
                    'pos': element[1],
                    'depth': element[2]}
                depthN.append(entry)
            else:
                startingPos = j
                break
        tree.append(depthN)
    
    return tree

def OLMUXMerge(L1, L2):
    """Merge two lists and sort merged list in ascending order

    Args:
        L1 (list): one of input list
        L2 (list): one of input list

    Returns:
        list: merged list
    """
    L = L1 + L2
    L.sort()
    return L

def OLMUXPackage(Qi):
    """Package

    Args:
        Qi (list): input list

    Returns:
        list: package list
    """
    outputList = []
    halfNum = int(len(Qi)/2)
    for j in range(halfNum):
        left = 2 * j
        right = left + 1
        sum = Qi[left][0] + Qi[right][0]
        outputList.append([sum, -1, left, right])
    return outputList

def OLMUXFunc(input, weight, muxTree, rns):
    """Processing

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        float: output of the FIR filter at one time point
    """    
    taps = len(input)
    # transform input to SC 
    input = np.transpose(input) 
    weight = np.transpose(weight)
    inputSC = np.logical_xor(rns[:, 0] < (input[i]+1)/2)
    sign = weight < 0
    productSC = np.logical_xor(inputSC, sign)
    
    # for i in range(taps):
    #     n1 = (productSC[i] == True).sum()
    #     n0 = (productSC[i] == False).sum()
    #     num = (n1-n0)/(n1+n0)
    #     print(f'product_{i}: {num}')

    """  
        which inputs connect to which ports of which MUXes of one level 
        eg. H = [5/32, 6/32, 7/32, 8/32, 16/32]
        level 1
        [
        [[1],[2]],
        [[3], [4]]
        ]
        level 2
        [
        [[1,2],[3,4]]
        ]
        level 3
        [
        [[1,2,3,4], [5]]
        ]
    """
    inputLinkToMuxes  = [] 
    # compute the results level by level. The output of previous level is the input of current level
    for i in range(len(muxTree)):
        # number of MUXes of current level. for first level, num of MUXes is len(Tree[0])/2 (len(Tree[0] must be a even number))
        # for later level, the number of MUXex is decided by the output of previous level and the number of primary input at that level. 
        muxNum = int((len(outputOfLastMuxes)+len(muxTree[i]))/2) if (i > 0) else int(len(muxTree[0])/2)
        input0 = np.empty((rns.shape[0],muxNum), dtype = bool)   # inputs to 0 port of all MUXes
        input1 = np.empty((rns.shape[0],muxNum), dtype = bool)   # inputs to 1 port of all MUXes
        s      = np.empty((rns.shape[0],muxNum), dtype = bool)   # input to selection port of all MUXes

        # input0Real = np.empty(muxNum, dtype = float)
        # input1Real = np.empty(muxNum, dtype = float)
        # sReal      = np.empty(muxNum, dtype = float)

        cnt    = 0     # how many MUXes have been created
        # deal with level after the first level
        if (i > 0):
            for j in range(len(outputOfLastMuxes)):
                if (j%2 == 0):
                    input0[:, cnt] = outputOfLastMuxes[:, j]
                    # n1 = (input0[cnt] == True).sum()
                    # n0 = (input0[cnt] == False).sum()
                    # num = (n1-n0)/(n1+n0)
                    # print(f'level_{i} mux_{cnt+1} port0 : {num}')

                    # input0Real[cnt] = outputOfLastMuxesReal[j]
                    # print(f'level_{i} mux_{cnt+1} port0_real : {input0Real[cnt]}')
                else:
                    input1[:, cnt] = outputOfLastMuxes[:, j]
                    # n1 = (input1[cnt] == True).sum()
                    # n0 = (input1[cnt] == False).sum()
                    # num = (n1-n0)/(n1+n0)
                    # print(f'level_{i} mux_{cnt+1} port1 :{num}')

                    # input1Real[cnt] = outputOfLastMuxesReal[j]
                    # print(f'level_{i} mux_{cnt+1} port1_real : {input1Real[cnt]}')
                    
                    cnt = cnt + 1
                    # merge input index
                    # since I deal with the output of MUX in order, so I always combine the first two entries in old inputLinkToMuxes
                    newLink = [inputLinkToMuxes[0][0]+inputLinkToMuxes[0][1], inputLinkToMuxes[1][0]+inputLinkToMuxes[1][1]]  
                    inputLinkToMuxes.append(newLink)
                    inputLinkToMuxes.pop(0) # discard first two old link
                    inputLinkToMuxes.pop(0)
                    
            if (len(muxTree[i]) > 0):
                    if (len(muxTree[i])%2 == 1):
                        input1[cnt] = productSC[muxTree[i][0]["pos"]]
                        newLink = [inputLinkToMuxes[0][0]+inputLinkToMuxes[0][1], [muxTree[i][0]["pos"]]]
                        inputLinkToMuxes.append(newLink)
                        inputLinkToMuxes.pop(0) # only discard once since only one link left
                        cnt = cnt + 1
                        
                        for j in range(1,len(muxTree[i])):
                            if (j % 2 == 1):        
                                input0[:, cnt] = productSC[:, muxTree[i][j]["pos"]]
                            else:
                                input1[:, cnt] = productSC[:, muxTree[i][j]["pos"]]   
                                cnt = cnt + 1
                                inputLinkToMuxes.append([[muxTree[i][j-1]["pos"]],[muxTree[i][j]["pos"]]]) 
                    else:
                        for j in range(len(muxTree[i])):
                            if (j % 2 == 0):        
                                input0[:, cnt] = productSC[:, muxTree[i][j]["pos"]]
                            else:
                                input1[:, cnt] = productSC[:, muxTree[i][j]["pos"]]   
                                cnt = cnt + 1
                                inputLinkToMuxes.append([[muxTree[i][j-1]["pos"]],[muxTree[i][j]["pos"]]]) 
                    
        # deal with first level
        else:
            for j in range(len(muxTree[0])):
                if (j%2 == 0):
                    input0[:, cnt] = productSC[:,muxTree[0][j]["pos"]]
                    # n1 = (input0[cnt] == True).sum()
                    # n0 = (input0[cnt] == False).sum()
                    # num = (n1-n0)/(n1+n0)
                    # print(f'level_{i} mux_{cnt+1} port0 :{num}')

                    # input0Real[cnt] = input[muxTree[0][j]["pos"]]
                    # print(f'level_{i} mux_{cnt+1} port0_real: {input0Real[cnt]}')
                else:
                    input1[:, cnt] = productSC[:,muxTree[0][j]["pos"]]
                    # n1 = (input1[cnt] == True).sum()
                    # n0 = (input1[cnt] == False).sum()
                    # num = (n1-n0)/(n1+n0)
                    # print(f'level_{i} mux_{cnt+1} port1 :{num}')

                    # input1Real[cnt] = input[Tree[0][j]["pos"]]
                    # print(f'level_{i} mux_{cnt+1} port1_real: {input1Real[cnt]}')

                    cnt = cnt + 1  
                    inputLinkToMuxes.append([[muxTree[0][j-1]["pos"]],[muxTree[0][j]["pos"]]])  # update
        
        # convert selection signal to SC 
        for j in range(len(inputLinkToMuxes)):
            sum0 = 0
            sum1 = 0
            # compute the sum of coefficients linking to port 0
            for k in range(len(inputLinkToMuxes[j][0])):
                sum0 = sum0 + abs(weight[inputLinkToMuxes[j][0][k]])

            # compute the sum of coefficients linking to port 1
            for k in range(len(inputLinkToMuxes[j][1])):
                sum1 = sum1 + abs(weight[inputLinkToMuxes[j][1][k]]) 

            c = sum1 / (sum0 + sum1)     
            s[:, j] =   rns[:, i] < c
            # sReal[j] = c
            # n1 = (s[j] == True).sum()
            # n0 = (s[j] == False).sum()
            # num = n1/(n1+n0)
            # print(f'level_{i} mux_{j} s: {num}')

            # print(f'level_{i} mux_{j} s_real: {c}')
        
        # call function to calculate the output of current depth
        outputOfLastMuxes = np.logical_or(np.logical_and(input0, np.invert(s)), np.logical_and(input1, s))
        #outputOfLastMuxesReal = depthLevelOutputReal(input0Real, input1Real, sReal)
    
    numOfTrue = (outputOfLastMuxes == True).sum()
    numOfFalse = (outputOfLastMuxes == False).sum()
    finalResult = (numOfTrue - numOfFalse) / (numOfTrue + numOfFalse)

    # trueResult = np.inner(np.array(H), np.array(input)) / (np.sum(abs(np.array(H))))
    # print(f'trueResult: {trueResult}')

    return finalResult
