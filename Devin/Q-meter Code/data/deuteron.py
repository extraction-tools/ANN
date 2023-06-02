import numpy as np
import cmath

gamm = 0.05
ncos2phi = 0.04

P = 0.9

pi = np.pi

def F1(R):
    epsilon=1
    x4 = (gamm*gamm + (1-(epsilon*R)-ncos2phi)*(1-(epsilon*R)-ncos2phi))
    x2 = cmath.sqrt(x4)
    x = cmath.sqrt(x2)
    y = cmath.sqrt(3-ncos2phi)
    y2 = y*y
    cos_alpha = (1-epsilon*R-ncos2phi)/x2
    alpha = cmath.acos(cos_alpha)
    
    term1 = 2*cmath.cos(alpha/2)*(cmath.atan((y2-x2)/(2*y*x*cmath.sin(alpha/2)))+(pi/2))
    subterm1 = y2 + x2 + 2*y*x*cmath.cos(alpha/2)
    subterm2 = y2 + x2 - 2*y*x*cmath.cos(alpha/2)
    term2 = cmath.sin(alpha/2)*cmath.log(subterm1/subterm2)
    return (1/(2*pi*x))*(term1+term2)

def F2(R):
    epsilon=-1
    x4 = (gamm*gamm + (1-(epsilon*R)-ncos2phi)*(1-(epsilon*R)-ncos2phi))
    x2 = cmath.sqrt(x4)
    x = cmath.sqrt(x2)
    y = cmath.sqrt(3-ncos2phi)
    y2 = y*y
    cos_alpha = (1-epsilon*R-ncos2phi)/x2
    alpha = cmath.acos(cos_alpha)
    
    term1 = 2*cmath.cos(alpha/2)*(cmath.atan((y2-x2)/(2*y*x*cmath.sin(alpha/2)))+(pi/2))
    subterm1 = y2 + x2 + 2*y*x*cmath.cos(alpha/2)
    subterm2 = y2 + x2 - 2*y*x*cmath.cos(alpha/2)
    term2 = cmath.sin(alpha/2)*cmath.log(subterm1/subterm2)
    return (1/(2*pi*x))*(term1+term2)

def generateDeuteronData(newP):
    P = newP

    fp = open("C:\\Users\\colin\\Desktop\\qmeter_simulation\\data\\DEUTERON.DAT","w")
    fp2 = open("C:\\Users\\colin\\Desktop\\qmeter_simulation\\data\\DDEUTERON.DAT","w")

    numpoints = 256
    calc_points = 100*numpoints

    #I_plus = []
    #I_minus = []
    I_combined = []
    x_vals = []

    left_bound = -6.0
    right_bound = 6.0

    delta = (right_bound - left_bound)/calc_points

    sqrtterm = cmath.sqrt(P*P - 4*(P-1)*(P+1))
    r = (-P - sqrtterm)/(2*(P-1))
    

    for i in range(calc_points):
        x = left_bound + i*delta
        x_vals.append(x)
        
        val_plus = F1(x)*cmath.sqrt(r)

        val_minus = F2(x)/cmath.sqrt(r)

        val_combined = val_plus+val_minus
        I_combined.append(val_combined.real)
        
    area = 0
    for i in range(calc_points-1):
        area += 0.5*delta*(I_combined[i]+I_combined[i+1])
    
    factor = P / area #this makes the total area proportional to the polarization
    
    scale_factor = 10 #this is used to make the signal larger so it is more visible in the simulation.
                        #In a real experiment this would be tied to the calibration constant
    
    
    for i in range(calc_points):
        I_combined[i] = I_combined[i] * factor * scale_factor
    
    return_vals = []
    for i in range(numpoints):
        val1 = I_combined[i*100]
        val2 = I_combined[i*100 + 1] 
        #print("Type: "+str(type(val1)))
        dval = (val2-val1)/delta
        
        fp.write(str(val1.real))
        fp2.write(str(val2.real))
        return_vals.append(val1.real)
    
        if (i != numpoints-1):
            fp.write("\n")
            fp2.write("\n")

    fp.close()
    fp2.close()
    
    return return_vals    

