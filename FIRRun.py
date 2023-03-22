import numpy as np
from math import *
from shared import *

def CVTRun(input, weight, rns):
    """Conventional design of a FIR filter

    Args:
        input (2d numpy array): inputs to the filter
        weight (1d numpy array): fiexed weight of the filter
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        1d numpy array with type float: output of the FIR filter
    """
    input = np.transpose(input)
    weight = np.reshape(weight, (1,-1))
    # Convert weight to bipolar represented SN
    weightSC = rns[:, 1] < (weight+1)/2
    # Compute unipolar represented selection signal
    selSC = rns[:, 2:] < 0.5

    result = np.empty(input.shape[0]) 
    for i in range(input.shape[0]):
        # Convert input to bipolar represented SN
        inputSC = rns[:, 0] < (input[i, :]+1)/2
        
        # Use xnor to perform multiplication of input and weight 
        productSC = np.logical_not(np.logical_xor(inputSC, weightSC))
        # Perform scaled addition
        outputSC = softMux(productSC, selSC)
        n1 = np.sum(outputSC)
        result[i] = 2*n1/rns.shape[0]-1

    return result


def HWARun(input, weight, height, rns):
    """Hard-wired weighted average(HWA) design of a FIR filter

    Args:
        input (2d numpy array): inputs to the filter
        weight (1d numpy array): fixed weight of the filter
        height (int): height of MUX
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        1d numpy array with type float: output of the FIR filter
    """
    input = np.transpose(input)
    weight = np.reshape(weight, (1,-1))
    # Normalized and quantized weight
    q = weightNormAndQuan(weight, height)
    
    selSC = rns[:, 1:] < 0.5
    sign = weight < 0

    signExt = np.zeros((1,2**height), dtype=bool)
    cnt = 0
    for j in range(len(q)):
            num = q[i]
            signExt[cnt:(cnt+num)] = sign[i, j]
            cnt = cnt + num

    result = np.empty(input.shape[0])
    for i in range(input.shape[0]):
        # Replicate input according to individual weight
        inputExt = np.empty((1,2**height))
        cnt = 0
        for j in range(len(q)):
            num = q[i]
            inputExt[cnt:(cnt+num)] = input[i, j]
            cnt = cnt + num

        # Convert input to bipolar represented SN
        inputSC = rns[:, 0] < (inputExt+1)/2
        
        productSC = np.logical_xor(inputSC, signExt)
        outputSC = softMux(productSC, selSC)
        
        n1 = np.sum(outputSC)
        result[i] = 2*n1/rns.shape[0]-1
    return result 

def MWARun(input, weight, rns):
    """Multi_level weighted average(MWA) design of a FIR filter

    Args:
        input (2d numpy array): inputs to the filter
        weight (1d numpy array): fixed weight of the filter
        rns (2d numpy array): used to convert input and weight to stochastic numbers

    Returns:
        1d numpy array with type float: output of the FIR filter
    """
    weight = np.reshape(weight, (1,-1))
    condProb = MWACalCondProb(np.abs(weight))
    input = np.transpose(input)
    
    # Transform conditional probability to SC numbers
    selSC = np.zeros((rns.shape[0], len(condProb)), dtype=bool)
    selSC[:, 0] = rns[:, 1] < (condProb[0][0] + 1)/2
    for i in range(1, len(condProb)):
        levelCondProb = np.array(condProb[i]).reshape(1,-1)
        muxInput = rns[:, 1+i] < (levelCondProb+1)/2

        selSC[:, i] = softMux(muxInput, selSC[:, i-1::-1])

    sign = weight < 0

    result = np.empty(input.shape[0])
    for i in range(len(input.shape[0])):
        # Transform input into SC numbers
        inputSC = rns[:, 0] < (input[i, :]+1)/2
        # Use xnor to perform multiplication of input and weight 
        productSC = np.logical_not(np.logical_xor(inputSC, sign))
        result[:, i] = softMux(productSC, selSC[:, ::-1])

    return result


def MWACalCondProb(weight):
    """Calculate conditional probabilities

    Args:
        weight (1d numpy array): weights of FIR filter

    Returns:
        list: list of conditional probabilties
    """
    scaling = np.sum(weight)
    jointProb = []
    numLevel = int(log2(len(weight)))
    # Calculate joint probability
    for i in range(numLevel):
        levelJointProb = []
        reshapedArray = np.reshape(weight,(len(weight)//2**(i+1), 2**(i+1)))
        for j in range(2**(i+1)):
            levelJointProb.append(np.sum(reshapedArray[:, j]) / scaling)
        jointProb.append(levelJointProb)
    
    # Calculate conditional probability
    condProb = [[jointProb[0][1]]]
    for i in range(1, len(jointProb)):
        levelCondProb = []
        for j in range(2**i,len(jointProb[i])):
            levelCondProb.append((jointProb[i][j])/jointProb[i-1][j-2**i])
        condProb.append(levelCondProb)
    
    return condProb
        

#def CeMuxRun(input, weight, height, rns):

def OLMUXRun(input, weight, rns):
    """Lowest optimum MUX tree design of a FIR filter

    Args:
        input (1d numpy array): input of the filter
        weight (1d numpy array): weight of the filter
        rns (1d numpy array): used to convert input and weight to stochastic numbers
    
    Returns:
        1d numpy array with type float: output of the FIR filter
    """
    # Decide input position in the optimum lowest tree
    inputTree = OLMUXCalInPos(weight)
    # Build optimum lowest tree
    muxTree = OLMUXBuildTree(inputTree)
    
    # transform input to SC 
    input = np.transpose(input) 
    weight = np.reshape(weight, (1,-1))
    inputSC = np.logical_xor(rns[:, 0] < (input[i]+1)/2)
    sign = weight < 0
    productSC = np.logical_xor(inputSC, sign)
    
    result = np.empty(input.shape[0])
    for i in range(input.shape[0]):
    # compute the results level by level. The output of previous level is the input of current level
        for j in range(len(muxTree)):
            cnt = 0 
            muxNum = len(muxTree[j]['selWeight'])
            input0 = np.empty((rns.shape[0],muxNum), dtype = bool)   # inputs to 0 port of all MUXes
            input1 = np.empty((rns.shape[0],muxNum), dtype = bool)   # inputs to 1 port of all MUXes
            s      = np.empty((rns.shape[0],muxNum), dtype = bool)   # input to selection port of all MUXes
            if (j==0):
                for k in range(len(muxTree[j]['primaryInput'])):
                    if (k%2==0):
                        input0[:, cnt] = productSC[muxTree[j]['primaryInput'][k]]
                    else:
                        input1[:, cnt] = productSC[muxTree[j]['primaryInput'][k]]
                        cnt = cnt + 1
            else:
                levelInput = np.empty((rns.shape[0], len(muxTree[j]['primaryInput'])), dtype = bool)
                for k in range(len(muxTree[j]['primaryInput'])):
                    levelInput[:, k] = productSC[muxTree[j]['primaryInput'][k]]
                input0 = np.concatenate((outputOfLastMuxes, levelInput), axis=1)[:, ::2]
                input1 = np.concatenate((outputOfLastMuxes, levelInput), axis=1)[:, 1::2]
            selArray = np.array([muxTree[j]['selWeight']])       
            s =   rns[:, j] < selArray
            # Calculate the output of current depth
            outputOfLastMuxes = np.logical_or(np.logical_and(input0, np.invert(s)), np.logical_and(input1, s))

    return result

def OLMUXCalInPos(weight):
    """Decide input position in the optimum lowest tree

    Args:
        weight (1d numpy array): array of weights of FIR filter

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

def OLMUXBuildTree(weight, inputTree):
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
    muxTree = []
    lastInputLinkToMuxes = []
    for i in range(len(inputTree)): 
        primaryInput = []
        level = {}
        muxSel = []
        inputLinkToMuxes = []
        linkMux = []
        sum0 = 0    
        sum1 = 0
        for j in range(len(lastInputLinkToMuxes)):
            linkMux = linkMux + lastInputLinkToMuxes[j]
            if (j % 2 == 0):
                for k in range(len(lastInputLinkToMuxes[j])):
                    sum0 = sum0 + abs(weight[lastInputLinkToMuxes[j][k]])
            else:
                inputLinkToMuxes.append(linkMux)
                linkMux = []
                for k in range(len(lastInputLinkToMuxes[j])):
                    sum1 = sum1 + abs(weight[lastInputLinkToMuxes[j][k]])
                muxSel.append(sum1/(sum0+sum1))
                sum0 = 0
                sum1 = 0

        if (len(inputTree[i])%2 == 0):
            for j in range(len(inputTree[i])):
                linkMux.append(inputTree[i][j]['pos'])
                primaryInput.append(inputTree[i][j]['pos'])
                if (j % 2 == 0):
                    sum0 = abs(weight[inputTree[i][j]["pos"]]) 
                else:
                    inputLinkToMuxes.append(linkMux)
                    linkMux = []
                    sum1 = abs(weight[inputTree[i][j]['pos']])
                    muxSel.append(sum1/(sum0+sum1))
                    sum0 = 0
                    sum1 = 0
            level['selWeight'] = muxSel
            level['primaryInput'] = primaryInput
            muxTree.append(level)   
            lastInputLinkToMuxes = inputLinkToMuxes
        else: 
            for j in range(len(inputTree[i])):
                linkMux.append(inputTree[i][j]['pos'])
                primaryInput.append(inputTree[i][j]['pos'])
                if (j % 2 == 0):
                    inputLinkToMuxes.append(linkMux)
                    linkMux = []
                    sum1 = abs(weight[inputTree[i][j]["pos"]]) 
                    muxSel.append(sum1/(sum0+sum1))
                    sum0 = 0
                    sum1 = 0
                else:
                    sum0 = abs(weight[inputTree[i][j]['pos']])
            level['selWeight'] = muxSel
            level['primaryInput'] = primaryInput
            muxTree.append(level)   
            lastInputLinkToMuxes = inputLinkToMuxes  

    return muxTree     

# weight = np.array([0.1, 0.2, -0.3, 0.4])
# print(MWACalCondProb(np.abs(weight)))