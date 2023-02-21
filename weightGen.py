import numpy as np

#-----input generation-----#
class weightGen():
    
    #-----generate weight-----#
    def gen(self, num):
        rng = np.random.default_rng()
        rn = rng.uniform(-1,1,num)

        return rn