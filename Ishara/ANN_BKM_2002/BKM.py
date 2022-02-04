# written by Annabel Li
# 10/16/2021
# complex numbers = tuple(re, im)
import math

# c and d are complex numbers
def cdstar(c, d):
    return ((c[0]*d[0]) - (c[1]*-1.0*d[1]), (c[0]*-1.0*d[1]) + (c[1]*d[0]))

# f is a complex number
def Rho2(f):
    return f[0]**2 + f[1]**2

# set constants
ALP_INV = 137.0359998
RAD = math.pi/180
M = 0.938272
M2 = M*M
GeV2nb = .389379*1000000

# set CFFs
H = (-0.897, 2.421)
E = (-0.541, 0.903)
Htilde = (2.444, 1.131)
Etilde = (0, 0)

# set kinematics
QQ = 1.82
x = 0.34
t = -0.17
k = 5.75
phi = 10 # in degrees
F1 = 0.685656
F2 = 1.09867

ee = 4. * M2 * x * x / QQ
y = math.sqrt(QQ) / ( math.sqrt(ee) * k )
xi = x * ( 1. + t / 2. / QQ ) / ( 2. - x + x * t / QQ )
Gamma1 = x * y * y / ALP_INV / ALP_INV / ALP_INV / math.pi / 8. / QQ / QQ / math.sqrt( 1. + ee )
s = 2. * M * k + M2
tmin = -1. * QQ * ( 2. * ( 1. - x ) * ( 1. - math.sqrt(1. + ee) ) + ee ) / ( 4. * x * ( 1. - x ) + ee )
K2 = - ( t / QQ ) * ( 1. - x ) * ( 1. - y - y * y * ee / 4.) * ( 1. - tmin / t ) * ( math.sqrt(1. + ee) + ( ( 4. * x * ( 1. - x ) + ee ) / ( 4. * ( 1. - x ) ) ) * ( ( t - tmin ) / QQ )  )

# BH lepton propagators
KD = - QQ / ( 2. * y * ( 1. + ee ) ) * ( 1. + 2. * math.sqrt(K2) * math.cos( math.pi - ( phi * RAD ) ) - t / QQ * ( 1. - x * ( 2. - y ) + y * ee / 2. ) + y * ee / 2. )
P1 = 1. + 2. * KD / QQ
P2 = t / QQ - 2. * KD / QQ

# BHUU
c0_BH = 8. * K2 * ( ( 2. + 3. * ee ) * ( QQ / t ) * ( F1 * F1  - F2 * F2 * t / ( 4. * M2 ) ) + 2. * x * x * ( F1 + F2 ) * ( F1 + F2 ) ) + ( 2. - y ) * ( 2. - y ) * ( ( 2. + ee ) * ( ( 4. * x * x * M2 / t ) * ( 1. + t / QQ ) * ( 1. + t / QQ ) + 4. * ( 1. - x ) * ( 1. + x * t / QQ ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 4. * x * x * ( x + ( 1. - x + ee / 2. ) * ( 1. - t / QQ ) * ( 1. - t / QQ ) - x * ( 1. - 2. * x ) * t * t / ( QQ * QQ ) ) * ( F1 + F2 ) * ( F1 + F2 ) ) + 8. * ( 1. + ee ) * ( 1. - y - ee * y * y / 4. ) * ( 2. * ee * ( 1. - t / ( 4. * M2 ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) - x * x * ( 1. - t / QQ ) * ( 1. - t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )
c1_BH = 8. * math.sqrt(K2) * ( 2. - y ) * ( ( 4. * x * x * M2 / t - 2. * x - ee ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 2. * x * x * ( 1. - ( 1. - 2. * x ) * t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )
c2_BH = 8. * x * x * K2 * ( ( 4. * M2 / t ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 2. * ( F1 + F2 ) * ( F1 + F2 ) )
Amp2_BH = 1. / ( x * x * y * y * ( 1. + ee ) * ( 1. + ee ) * t * P1 * P2 ) * ( c0_BH + c1_BH * math.cos( math.pi - (phi * RAD) ) + c2_BH * math.cos( 2. * ( math.pi - ( phi * RAD ) ) )  )
Amp2_BH = GeV2nb * Amp2_BH
dsigma_BH = Gamma1 * Amp2_BH

# DVCSUU
c_dvcs = 1./(2. - x)/(2. - x) * ( 4. * ( 1 - x ) * ( Rho2(H) + Rho2(Htilde) ) - x * x * ( cdstar(H, E)[0] + cdstar(E, H)[0] + cdstar(Htilde, Etilde)[0] + cdstar(Etilde, Htilde)[0] ) - ( x * x + (2. - x) * (2. - x) * t / 4. / M2 ) * Rho2(E) - ( x * x * t / 4. / M2 ) * Rho2(Etilde) )
c0_dvcs = 2. * ( 2. - 2. * y + y * y ) * c_dvcs
Amp2_DVCS = 1. / ( y * y * QQ ) *  c0_dvcs
Amp2_DVCS = GeV2nb * Amp2_DVCS
dsigma_DVCS = Gamma1 * Amp2_DVCS

# IUU
A = - 8. * K2 * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * ( 2. - y ) * ( 1. - y ) * ( 2. - x ) * t / QQ - 8. * math.sqrt(K2) * ( 2. - 2. * y + y * y ) * math.cos( math.pi - (phi * RAD) )
B = 8. * x * x * ( 2. - y ) * (1 - y ) / ( 2. - x ) * t / QQ
C =  x / ( 2. - x ) * ( - 8. * K2 * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * math.sqrt(K2) * ( 2. - 2. * y + y * y ) * math.cos( math.pi - (phi * RAD) ) )
Amp2_I = 1. / ( x * y * y * y * t * P1 * P2 ) * ( A * ( F1 * H[0] - t / 4. / M2 * F2 * E[0] ) + B * ( F1 + F2 ) * ( H[0] + E[0] ) + C * ( F1 + F2 ) * Htilde[0] )
Amp2_I = GeV2nb * Amp2_I
dsigma_I = Gamma1 * Amp2_I

F = dsigma_BH + dsigma_DVCS + dsigma_I
print(dsigma_BH)
print(dsigma_DVCS)
print(dsigma_I)
print(F) 