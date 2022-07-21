import numpy as np
import pandas as pd

######################################################################################
########### User Input ########################################
Formalism = 'BKM10'

# Input kinematics from a file (use the proper file name):
Input_Kinematics_and_CFFs ='BKM02_pseudodata_with_true_CFFs.csv'

# Input kinematics and CFFs using a model
arr_k = np.array([5.8, 5.8, 5.8, 5.8])
arr_qq = np.array([1.82, 1.93, 1.96, 1.98])
arr_xb = np.array([0.34, 0.36, 0.37, 0.39])
arr_t = np.array([-0.17, -0.23, -0.27, -0.32])
n_phi = 45 # number of phi in one set

smear_err = 0.05   # This factor smears the F to generate its uncertainty
###############################################################

if (Formalism == 'BKM02'):
    from BHDVCS_formalism import BKM02
    dv = BKM02()
elif (Formalism == 'BKM10'):
    from BHDVCS_formalism import BKM10
    dv = BKM10()
######################################################################################



###############################################
#### Generating with kinematics using ranges ##
## (use this next section and comment next  ###
## section (commented by default)           ###
## to use kinematics from a .csv file #########
###############################################

#func_fit = dv.curve_fit2
#f = dv.plot_fit

F1_term = dv.Get_F1
F2_term = dv.Get_F2

Set = np.array([])
ind = np.array([])
phi = np.array([])
k = np.array([])
qq = np.array([])
xb = np.array([])
t = np.array([])
gReH = np.array([])
gReE = np.array([])
gReHTilde = np.array([])
gdvcs = np.array([])

F1 = np.array([])
F2 = np.array([])

for ii in range(arr_k.size):
  for jj in range(n_phi):
    Set = np.append(Set, ii+1)
    ind = np.append(ind,jj)
    phi = np.append(phi, 360.0/n_phi*jj + 0.5 * 360/n_phi)
    k = np.append(k, arr_k[ii])
    qq = np.append(qq, arr_qq[ii])
    xb = np.append(xb, arr_xb[ii])
    t = np.append(t, arr_t[ii])
    
F1 = F1_term(t)
F2 = F2_term(t)
#Input model and smearing error
gReH = 5*t*t + 2*xb*xb
gReE =  4.5*t - 1.5*xb*xb
gReHTilde = -5.5*t + 4.5*xb
gdvcs = xb/10.


#######################################################################################
### Uncomment the following section if user needs to take kinematics from an already ##
###################### .csv generated file     ########################################
#######################################################################################

#F1_term = dv.Get_F1
#F2_term = dv.Get_F2
#dats = pd.read_csv(Input_Kinematics_and_CFFs)
#Set = np.array(dats['#Set'])
#ind = np.array(dats['index'])
#k = np.array(dats['k'])
#qq = np.array(dats['QQ'])
#xb = np.array(dats['x_b'])
#t = np.array(dats['t'])
#phi = np.array(dats['phi_x'])
#F = np.array(dats['F'])
#errF = np.array(dats['sigmaF'])
#F1 = F1_term(t)
#F2 = F2_term(t)
#gReH = np.array(dats['ReH'])
#gReE = np.array(dats['ReE'])
#gReHTilde = np.array(dats['ReHTilde'])
#gdvcs = np.array(dats['dvcs'])

######################################################################################
########            Calculating F       ##############################################
#######################################################################################

# Calculating F #
f_xdat = np.array([phi, qq, xb, t, k, F1, F2])
if (Formalism == 'BKM02'):
    func_fit = dv.curve_fit2
    f_truth = func_fit(f_xdat, gReH, gReE, gReHTilde, gdvcs)
elif (Formalism == 'BKM10'):
    func_fit = dv.curve_fit2
    f_truth = func_fit(f_xdat, gReH, gReE, gReHTilde, gdvcs)

# Smearing F #
F_replica = np.random.normal(f_truth, smear_err*f_truth)
F = F_replica
err_smear =  smear_err * f_truth
dat = np.array([Set, ind, k, qq, xb, t, phi, F, err_smear,  F1, F2, gReH, gReE, gReHTilde, gdvcs])
dat = dat.T
Form = np.array([str(Formalism) for i in range(0,len(F))])

#######################################################################################

data_dictionary = {"Formalism": [],"#Set":[], "index":[], "k":[], "QQ":[], "t":[], "phi_x":[], "F1":[], "F2":[], "ReH_true":[], "ReE_true":[], "ReHTilde_true":[], "c0_true":[], "F":[], "sigmaF":[]}

data_dictionary["#Set"] = Set
data_dictionary["index"] = ind
data_dictionary["k"] = k
data_dictionary["QQ"] = qq
data_dictionary["t"] = t
data_dictionary["phi_x"] = phi
data_dictionary["F1"] = F1
data_dictionary["F2"] = F2
data_dictionary["ReH"] = gReH
data_dictionary["ReE"] = gReE
data_dictionary["ReHTilde"] = gReHTilde
data_dictionary["c0_true"] = gdvcs
data_dictionary["F"] = F
data_dictionary["sigmaF"] = err_smear
data_dictionary["Formalism"] = Form

df = pd.DataFrame(data_dictionary)
df.to_csv(str(Formalism)+'_'+'pseudodata_generation.csv')


