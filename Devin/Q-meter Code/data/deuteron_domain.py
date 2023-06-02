import numpy as np


def generateDomain(newP):
    ints = range(0,256)
    k_ints = np.array(ints, float)
    domain = ((k_ints / 256)*12)-6.0
    return domain
    
