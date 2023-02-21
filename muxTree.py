from math import *
from operator import mod, xor
import numpy as np

class muxTree():

    #-----constructor-----#
    # parameter
    #   H: list of coefficient
    def __init__(self, H):
        self.H = np.array(H)  # coefficients
        
        # structure of MUX tree
        # eg. H = [5/32, 6/32, 7/32, 8/32, 16/32] 
        # [
        #   [{"value":H0, "pos":0, "depth":3}, {"value":H1, "pos":1, "depth":3}, {"value":H2, "pos":2, "depth":3}, {"value":H3, "pos":3, "depth":3}],
        #                                      [                                                                ],
        #                                                      [{"value":H4, "pos":4, "depth":1}]y
        # ]
        self.Tree = []  
        
        if (len(H) == 0):
            self.order = 0 # order of FIR
        else:
            self.order = len(H)

    #-----build optimum MUX tree-----#
    def buildTree(self):
        D = ceil(log2(self.order))  # height of MUX tree
        Q = [] # packages 
        depth = []  # depth of all the coefficients
        for i in range(self.order):
            depth.append([self.H[i],i, 0])  # Initialization 
        Q0 = [] # Q0 is a list
        for i in range(self.order):
            #[H[i], i, -1, -1], first entry: value, second entry: position in the given coeffcients list, third entry: position of left child in the previous package, fourth entry: position of right child in the previous package
            Q0.append([abs(self.H[i]), i, -1, -1])    
        Q0.sort()
        Q.append(Q0)    # Q is a list of list
        
        # len of Q: D
        for i in range(D-1):
            Q.append(self.merge(Q[0],self.package(Q[i])))   # calculate all packages
        
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
        startingPos = 0 # since depth list is already sorted, recording the starting postion of each layer makes the construction efficient.
        for i in range(D, 0, -1):
            depthN = []
            for j in range(startingPos, self.order):
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
            self.Tree.append(depthN)

    #-----merge two lists and sort merged list in ascending order-----#
    # parameter
    #   L1, L2: input lists
    # return
    #   L: merged and sorted list
    def merge(self, L1, L2):
        L = L1 + L2
        L.sort()
        return L

    #-----package-----#
    # parameter
    #   Qi: list
    # return
    #   outputList: package list
    def package(self, Qi):
        outputList = []
        halfNum = int(len(Qi)/2)
        for j in range(halfNum):
            left = 2 * j
            right = left + 1
            sum = Qi[left][0] + Qi[right][0]
            outputList.append([sum, -1, left, right])
        return outputList
    
    
    #-----run-----#
    # parameter
    #   input: input
    #   RNS: array of random number
    # return
    #   sc-based filter result
    def run(self, input, RNS):
        input = np.array(input)
        result = np.empty(len(input)+len(self.H)-1)

        for i in range(len(input)+len(self.H)-1):
            inputOnce = np.zeros(len(self.H))
            if (i < len(self.H)-1):
                for j in range(i+1):
                    inputOnce[j] = input[i-j]
            elif (i < len(input)):
                for j in range(len(self.H)):
                    inputOnce[j] = input[i-j]
            else:
                for j in range(i-len(input)+1, len(self.H)):
                    inputOnce[j] = input[i-j]
            result[i] = self.runOnce(inputOnce, RNS) * (np.sum(abs(self.H)))
        
        return result

    #-----run once-----#
    # parameter
    #   input: input
    #   RNS: array of random number
    # return
    #   finalResult: result of one input
    def runOnce(self, input, RNS):
        input = np.array(input)
        RNS = np.array(RNS)
        # transform input to SC 
        # I use bipolar SC representation here. p = (1+x)/2, which means that the reference value should be (1+x)/2
        inputSC = np.empty((self.order, RNS.shape[1]), dtype = bool)    
        for i in range(self.order):
            inputSC[i] = xor((RNS[0] < (input[i]+1)/2), (self.H[i] < 0))   # output of XOR gate, one input of XOR gate is sc sequence of input
                                                                            # another input is sign of correspoding coefficient
            n1 = (inputSC[i] == True).sum()
            n0 = (inputSC[i] == False).sum()
            num = (n1-n0)/(n1+n0)
            print(f'input_{i}: {num}')
        
        # which inputs connect to which ports of which MUXes of one level 
        # eg. H = [5/32, 6/32, 7/32, 8/32, 16/32]
        # level 1
        # [
        #   [[1],[2]],
        #   [[3], [4]]
        # ]
        # level 2
        #[
        #   [[1,2],[3,4]]
        # ]
        # level 3
        #[
        #   [[1,2,3,4], [5]]
        # ]
        inputLinkToMuxes  = [] 
        # compute the results level by level. The output of previous level is the input of current level
        for i in range(len(self.Tree)):
            # number of MUXes of current level. for first level, num of MUXes is len(self.Tree[0])/2 (len(self.Tree[0] must be a even number))
            # for later level, the number of MUXex is decided by the output of previous level and the number of primary input at that level. 
            muxNum = int((len(outputOfLastMuxes)+len(self.Tree[i]))/2) if (i > 0) else int(len(self.Tree[0])/2)
            input0 = np.empty((muxNum, RNS.shape[1]), dtype = bool)   # inputs to 0 port of all MUXes
            input1 = np.empty((muxNum, RNS.shape[1]), dtype = bool)   # inputs to 1 port of all MUXes
            s      = np.empty((muxNum, RNS.shape[1]), dtype = bool)   # Input to selection port of all MUXes

            input0Real = np.empty(muxNum, dtype = float)
            input1Real = np.empty(muxNum, dtype = float)
            sReal      = np.empty(muxNum, dtype = float)

            cnt    = 0     # how many MUXes have been created
            # deal with level after the first level
            if (i > 0):
                for j in range(len(outputOfLastMuxes)):
                    if (mod(j,2) == 0):
                        input0[cnt] = outputOfLastMuxes[j]
                        n1 = (input0[cnt] == True).sum()
                        n0 = (input0[cnt] == False).sum()
                        num = (n1-n0)/(n1+n0)
                        print(f'level_{i} mux_{cnt+1} port0 : {num}')

                        input0Real[cnt] = outputOfLastMuxesReal[j]
                        print(f'level_{i} mux_{cnt+1} port0_real : {input0Real[cnt]}')
                    else:
                        input1[cnt] = outputOfLastMuxes[j]
                        n1 = (input1[cnt] == True).sum()
                        n0 = (input1[cnt] == False).sum()
                        num = (n1-n0)/(n1+n0)
                        print(f'level_{i} mux_{cnt+1} port1 :{num}')

                        input1Real[cnt] = outputOfLastMuxesReal[j]
                        print(f'level_{i} mux_{cnt+1} port1_real : {input1Real[cnt]}')
                        
                        cnt = cnt + 1
                        # merge input index
                        # since I deal with the output of MUX in order, so I always combine the first two entries in old inputLinkToMuxes
                        newLink = [inputLinkToMuxes[0][0]+inputLinkToMuxes[0][1], inputLinkToMuxes[1][0]+inputLinkToMuxes[1][1]]  
                        inputLinkToMuxes.append(newLink)
                        inputLinkToMuxes.pop(0) # discard first two old link
                        inputLinkToMuxes.pop(0)
                        
                if (len(self.Tree[i]) != 0):
                    if (len(self.Tree[i]) == 1):
                        input1[cnt] = inputSC[self.Tree[i][0]["pos"]]
                        n1 = (input0[cnt] == True).sum()
                        n0 = (input0[cnt] == False).sum()
                        num = (n1-n0)/(n1+n0)
                        print(f'level_{i} mux_{cnt+1} port1 :{num}')

                        input1Real[cnt] = input[self.Tree[i][0]["pos"]]
                        print(f'level_{i} mux_{cnt+1} port1_real : {input1Real[cnt]}')

                        newLink = [inputLinkToMuxes[0][0]+inputLinkToMuxes[0][1], [self.Tree[i][0]["pos"]]]
                        inputLinkToMuxes.append(newLink)
                        inputLinkToMuxes.pop(0) # only discard once since only one link left
                    else:
                        for j in range(len(self.Tree[i])):
                            if (mod(j,2) == 0):
                                input0[cnt] = inputSC[self.Tree[i][j]["pos"]]
                                n1 = (input0[cnt] == True).sum()
                                n0 = (input0[cnt] == False).sum()
                                num = (n1-n0)/(n1+n0)
                                print(f'level_{i} mux_{cnt+1} port0 :{num}')

                                input0Real[cnt] = input[self.Tree[i][j]["pos"]]
                                print(f'level_{i} mux_{cnt+1} port0_real : {input0Real[cnt]}')
                        else:
                            input1[cnt] = inputSC[self.Tree[i][j]["pos"]]
                            n1 = (input1[cnt] == True).sum()
                            n0 = (input1[cnt] == False).sum()
                            num = (n1-n0)/(n1+n0)
                            print(f'level_{i} mux_{cnt+1} port1 :{num}')

                            input1Real[cnt] = input[self.Tree[i][j]["pos"]]
                            print(f'level_{i} mux_{cnt+1} port1_real : {input1Real[cnt]}')
                            cnt = cnt + 1
                            inputLinkToMuxes.append([[self.Tree[i][j-1]["pos"]],[self.Tree[i][j]["pos"]]])
                        
            # deal with first level
            else:
                for j in range(len(self.Tree[0])):
                    if (mod(j,2) == 0):
                        input0[cnt] = inputSC[self.Tree[0][j]["pos"]]
                        n1 = (input0[cnt] == True).sum()
                        n0 = (input0[cnt] == False).sum()
                        num = (n1-n0)/(n1+n0)
                        print(f'level_{i} mux_{cnt+1} port0 :{num}')

                        input0Real[cnt] = input[self.Tree[0][j]["pos"]]
                        print(f'level_{i} mux_{cnt+1} port0_real: {input0Real[cnt]}')
                    else:
                        input1[cnt] = inputSC[self.Tree[0][j]["pos"]]
                        n1 = (input1[cnt] == True).sum()
                        n0 = (input1[cnt] == False).sum()
                        num = (n1-n0)/(n1+n0)
                        print(f'level_{i} mux_{cnt+1} port1 :{num}')

                        input1Real[cnt] = input[self.Tree[0][j]["pos"]]
                        print(f'level_{i} mux_{cnt+1} port1_real: {input1Real[cnt]}')

                        cnt = cnt + 1  
                        inputLinkToMuxes.append([[self.Tree[0][j-1]["pos"]],[self.Tree[0][j]["pos"]]])  # update
            
            # convert selection signal to SC 
            for j in range(len(inputLinkToMuxes)):
                sum0 = 0
                sum1 = 0
                # compute the sum of coefficients linking to port 0
                for k in range(len(inputLinkToMuxes[j][0])):
                    sum0 = sum0 + abs(self.H[inputLinkToMuxes[j][0][k]])

                # compute the sum of coefficients linking to port 1
                for k in range(len(inputLinkToMuxes[j][1])):
                    sum1 = sum1 + abs(self.H[inputLinkToMuxes[j][1][k]]) 

                c = sum1 / (sum0 + sum1)     
                s[j] =   RNS[i] < c
                sReal[j] = c
                n1 = (s[j] == True).sum()
                n0 = (s[j] == False).sum()
                num = n1/(n1+n0)
                print(f'level_{i} mux_{j} s: {num}')

                print(f'level_{i} mux_{j} s_real: {c}')
            
            # call function to calculate the output of current depth
            outputOfLastMuxes     = self.depthLevelOutput(input0, input1, s)
            outputOfLastMuxesReal = self.depthLevelOutputReal(input0Real, input1Real, sReal)
        
        numOfTrue = (outputOfLastMuxes == True).sum()
        numOfFalse = (outputOfLastMuxes == False).sum()
        finalResult = (numOfTrue - numOfFalse) / (numOfTrue + numOfFalse)

        trueResult = np.inner(np.array(self.H), np.array(input)) / (np.sum(abs(np.array(self.H))))
        print(f'trueResult: {trueResult}')

        return finalResult
    
    #-----output of each depth-----#
    # parameter
    #   i0: input to the 0 port of MUX
    #   i1: input to the 1 port of MUX
    #   s : input to the selection port of mux
    # output
    #   outputSequence: output of MUXes
    def depthLevelOutput(self, i0, i1, s):
        i0 = np.array(i0)
        i1 = np.array(i1)
        s  = np.array(s)
        outputSequence = np.empty(i0.shape, dtype=i0.dtype)
        resultArray = np.empty(i0.shape[0], dtype = float)
        for i in range(i0.shape[0]):
            outputSequence[i] = np.logical_or(np.logical_and(i0[i], np.invert(s[i])), np.logical_and(i1[i], s[i]))  # output of one MUX  
            n1 = (outputSequence[i] == True).sum()
            n0 = (outputSequence[i] == False).sum()
            num = (n1-n0)/(n1+n0)
            resultArray[i] = num
        
        print(f'mux_output: {resultArray}')
        return outputSequence

    #-----real output of each depth-----#
    # parameter
    #   i0: input to the 0 port of MUX
    #   i1: input to the 1 port of MUX
    #   s : input to the selection port of mux
    # output
    #   outputSequence: output of MUXes
    def depthLevelOutputReal(self, i0, i1, s):
        i0 = np.array(i0)
        i1 = np.array(i1)
        s  = np.array(s)
        
        outputSequence = s*i1 + (1-s)*i0  
        print(f'mux_output_real: {outputSequence}')
        return outputSequence

