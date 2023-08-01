import tensorly as tl
import numpy as np
import matplotlib.pyplot as plt

# +
M = 1.
Q = 1.
Gamma = 1.
y = 1.
xB = 1.
k0 = float(Q/(Gamma*y))
t = 1.
theta_prime = theta = 2*np.pi 
phi = np.pi
Pmag = np.sqrt(-t*(1 - (t/4*M*M)))
k0_prime = Q*(1-y)/(Gamma*y)
q0_prime = (Q*Q + xB*t)/(2*M*xB)
cos_theta = -(1+((Gamma*Gamma)/(2))*((Q*Q + t)/(Q*Q + xB*t)))/(np.sqrt(1 + Gamma*Gamma))
cos_theta_prime = -(Gamma*Gamma*(Q*Q - t) - 2*xB*t)/(4*xB*M*np.sqrt(1+Gamma*Gamma)*np.sqrt(-t*((1-(t)/(4*M*M)))))
cos_theta_l = -(1+y*(Gamma*Gamma)/2)/(np.sqrt(1+Gamma*Gamma))
cos_theta_l_prime = -(1-y-(Gamma*Gamma*y)/2)/((1-y)*np.sqrt(1+Gamma*Gamma))
sin_theta_l = ((Gamma)/(np.sqrt(1+Gamma*Gamma)))*np.sqrt(1-y-(y*y*Gamma*Gamma)/(4))
sin_theta_l_prime = sin_theta_l/(1-y)




# +
P = tl.vec_to_tensor([M,0,0,0],shape = (4,1))
q = tl.vec_to_tensor([Q/Gamma,0,0,-(Q/Gamma)*np.sqrt(1+Gamma*Gamma)],shape = (4,1))
k = tl.vec_to_tensor([k0,k0*sin_theta_l,0,k0*cos_theta_l],shape = (4,1))
k_prime = tl.vec_to_tensor([k0_prime,k0_prime*sin_theta_l_prime,0,k0_prime*cos_theta_l_prime],shape = (4,1))
P_prime = tl.vec_to_tensor([M - (t/(2*M)), Pmag*np.sin(theta_prime)*np.cos(phi),Pmag*np.sin(theta_prime)*np.sin(phi),Pmag*cos_theta_l],shape = (4,1))
q_prime = tl.vec_to_tensor([q0_prime,q0_prime*np.sin(theta)*np.cos(phi),k0_prime*np.sin(theta)*np.sin(phi),q0_prime*np.cos(theta)], shape = (4,1))

q[...,0]
# -


