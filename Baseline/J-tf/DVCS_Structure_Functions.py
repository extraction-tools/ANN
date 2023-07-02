import math
import numpy as np
import tensorflow as tf

twist = 2

class JI_DVCS(object):
    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998)  # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535)
        self.RAD = tf.constant(self.PI / 180.)
        self.M = tf.constant(0.938272)  # Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379 * 1000000)  # Conversion from GeV to NanoBar
        self.M2 = tf.constant(0.938272 * 0.938272)  # Mass of the proton  squared in GeV

    @tf.function
    def hampc(self, q, k, p, xi):
        if twist == 2:
            return (q * q ( k * q * p) * (k * q * p) + (k * q) * ((q * q * p)*(q * q * p) + (q * p) * (q * p) * (k * q - q * q)) - 2 * (k * q) * (k * q * p) * (q * q * p)) / (q * q * (q * p) * (q * p) - (q * (q * p)) * (q * (q * p)))
        elif twist == 3:
            return 
        
    def htampc(self, q, k, p, xi):
        if twist == 2:
            return (q * q ( k * q * p) * (k * q * p) + (k * q) * ((q * q * p)*(q * q * p) + (q * p) * (q * p) * (k * q - q * q)) - 2 * (k * q) * (k * q * p) * (q * q * p)) / (q * q * (q * p) * (q * p) - (q * (q * p)) * (q * (q * p)))
        elif twist == 3:
            
            return 
        
    def hplusampc(self):
        if twist ==2:
            return 0
        elif twist == 3:
            return 
    
    def hminusampc(self,q,k,p,xi):
        if twist == 2:
            return ( q * q (k * q * p) - (k * q) * (q * q * p)) / np.sqrt((q * q * p) * (q * q * p) - q * q * (q * p) * (q * p))
    