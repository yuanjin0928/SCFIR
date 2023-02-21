import numpy as np
from shared import *

#-----implementation of multi-level weigted average (MWA) design-----#
class MWAFIR():
    
    def __init__(self):
        self.input = None
        self.weight = None

    def getInput(self, input):
        self.input = np.array(input)

    def getWeight(self, weight):
        self.weight = np.array(weight)

    
