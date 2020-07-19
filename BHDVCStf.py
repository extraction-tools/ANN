
import tensorflow as tf
import Lorentz_Vector as lv

class BHDVCS(tf.Module):

    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998, dtype=tf.float64) # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535, dtype=tf.float64)
        self.RAD = tf.constant(self.PI / 180., dtype=tf.float64)
        self.M = tf.constant(0.938272, dtype=tf.float64) #Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379*1000000, dtype=tf.float64) # Conversion from GeV to NanoBarn
        # Elastic FF
        # self.F1  # Dirac FF - helicity conserving (non spin flip)
        # self.F2  # Pauli FF - helicity non-conserving (spin flip)

        # self.QQ, x, t, k# creo q no hace falta
        # self.y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau

        # 4-momentum vectors
        self.K = lv.LorentzVector()
        self.KP = lv.LorentzVector()
        self.Q = lv.LorentzVector()
        self.QP  = lv.LorentzVector()
        self.D  = lv.LorentzVector()
        self.p  = lv.LorentzVector()
        self.P  = lv.LorentzVector()
        # 4 - vector products independent of phi
        self.kkp = 0
        self.kq =0
        self.kp =0
        self.kpp=0
        # 4 - vector products dependent of phi
        # self.kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd

        #     #self.KK_T, KQP_T, KKP_T, KXQP_T, KD_T, DD_T
        # self.kk_T, kqp_T, kkp_T, kd_T, dd_T

        # self.s     # Mandelstam variable s which is the center of mass energy
        # self.Gamma # Factor in front of the cross section
        # self.jcob  #Defurne's Jacobian

        # self.AUUBH, BUUBH # Coefficients of the BH unpolarized structure function FUU_BH
        # self.AUUI, BUUI, CUUI # Coefficients of the BHDVCS interference unpolarized structure function FUU_I
        # self.con_AUUBH, con_BUUBH, con_AUUI, con_BUUI, con_CUUI  # Coefficients times the conversion to nb and the jacobian
        # self.bhAUU, bhBUU # Auu and Buu term of the BH cross section
        # self.iAUU, iBUU, iCUU # Terms of the interference containing AUUI, BUUI and CUUI
        # self.xbhUU # Unpolarized BH cross section
        # self.xIUU # Unpolarized interference cross section

    def TProduct(self, v1, v2):
        return v1.Px()*v2.Px() + v1.Py()*v2.Py()

    def SetKinematics(self, _QQ, _x, _t, _k):

        self.QQ = _QQ #Q^2 value
        self.x = _x   #Bjorken x
        self.t = _t   #momentum transfer squared
        self.k = _k   #Electron Beam Energy
        self.M2 = self.M*self.M #Mass of the proton  squared in GeV
        #fractional energy of virtual photon
        self.y = self.QQ / ( 2. * self.M * self.k * self.x ) # From eq. (23) where gamma is substituted from eq (12c)
        #squared gamma variable ratio of virtuality to energy of virtual photon
        self.gg = 4. * self.M2 * self.x * self.x / self.QQ # This is gamma^2 [from eq. (12c)]
        #ratio of longitudinal to transverse virtual photon flux
        self.e = ( 1 - self.y - ( self.y * self.y * (self.gg / 4.) ) ) / ( 1. - self.y + (self.y * self.y / 2.) + ( self.y * self.y * (self.gg / 4.) ) ) # epsilon eq. (32)
        #Skewness parameter
        self.xi = 1. * self.x * ( ( 1. + self.t / ( 2. * self.QQ ) ) / ( 2. - self.x + self.x * self.t / self.QQ ) ) # skewness parameter eq. (12b) dnote: there is a minus sign on the write up that shouldn't be there
        #Minimum t value
        self.tmin = ( self.QQ * ( 1. - tf.sqrt( 1. + self.gg ) + self.gg / 2. ) ) / ( self.x * ( 1. - tf.sqrt( 1. + self.gg ) + self.gg / ( 2.* self.x ) ) ) # minimum t eq. (29)
        #Final Lepton energy
        self.kpr = self.k * ( 1. - self.y ) # k' from eq. (23)
        #outgoing photon energy
        self.qp = self.t / 2. / self.M + self.k - self.kpr #q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
        #Final proton Energy
        self.po = self.M - self.t / 2. / self.M # This is p'_0 from eq. (28b)
        self.pmag = tf.sqrt( ( -1*self.t ) * ( 1. - (self.t / (4. * self.M *self.M ))) ) # p' magnitude from eq. (28b)
        #Angular Kinematics of outgoing photon
        self.cth = -1. / tf.sqrt( 1. + self.gg ) * ( 1. + self.gg / 2. * ( 1. + self.t / self.QQ ) / ( 1. + self.x * self.t / self.QQ ) ) # This is tf.cos(theta) eq. (26)
        self.theta = tf.acos(self.cth) # theta angle
        #print('Theta: ', self.theta)
        #Lepton Angle Kinematics of initial lepton
        self.sthl = tf.sqrt( self.gg ) / tf.sqrt( 1. + self.gg ) * ( tf.sqrt ( 1. - self.y - self.y * self.y * self.gg / 4. ) ) # sin(theta_l) from eq. (22a)
        self.cthl = -1. / tf.sqrt( 1. + self.gg ) * ( 1. + self.y * self.gg / 2. )  # tf.cos(theta_l) from eq. (22a)
        #ratio of momentum transfer to proton mass
        self.tau = -0.25 * self.t / self.M2

        # phi independent 4 - momenta vectors defined on eq. (21) -------------
        self.K.SetPxPyPzE( self.k * self.sthl, 0.0, self.k * self.cthl, self.k )
        self.KP.SetPxPyPzE( self.K[0], 0.0, self.k * ( self.cthl + self.y * tf.sqrt( 1. + self.gg ) ), self.kpr )
        self.Q = self.K - self.KP
        self.p.SetPxPyPzE(0.0, 0.0, 0.0, self.M)

        # Sets the Mandelstam variable s which is the center of mass energy
        self.s = (self.p + self.K) * (self.p + self.K)

        # The Gamma factor in front of the cross section
        self.Gamma = 1. / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / self.PI / 16. / ( self.s - self.M2 ) / ( self.s - self.M2 ) / tf.sqrt( 1. + self.gg ) / self.x

        # Defurne's Jacobian
        self.jcob = 1./ ( 2. * self.M * self.x * self.K[3] ) * 2. * self.PI * 2.
        #print("Jacobian: ", self.jcob)
        #___________________________________________________________________________________

    def Set4VectorsPhiDep(self, phi) :

        # phi dependent 4 - momenta vectors defined on eq. (21) -------------

        self.QP.SetPxPyPzE(self.qp * tf.sin(self.theta) * tf.cos( phi * self.RAD ), self.qp * tf.sin(self.theta) * tf.sin( phi * self.RAD ), self.qp * tf.cos(self.theta), self.qp)
        self.D = self.Q - self.QP # delta vector eq. (12a)
        #print(self.D, "\n", self.Q, "\n", self.QP)
        self.pp = self.p + self.D # p' from eq. (21)
        self.P = self.p + self.pp
        self.P.SetPxPyPzE(.5*self.P.Px(), .5*self.P.Py(), .5*self.P.Pz(), .5*self.P.Pt())

        #____________________________________________________________________________________

    def Set4VectorProducts(self, phi) :

        # 4-vectors products (phi - independent)
        self.kkp  = self.K * self.KP   #(kk')
        self.kq   = self.K * self.Q    #(kq)
        self.kp   = self.K * self.p    #(pk)
        self.kpp  = self.KP * self.p   #(pk')

        # 4-vectors products (phi - dependent)
        self.kd   = self.K * self.D    #(kd)
        self.kpd  = self.KP * self.D   #(k'd)
        self.kP   = self.K * self.P    #(kP)
        self.kpP  = self.KP * self.P   #(k'P)
        self.kqp  = self.K * self.QP   #(kq')
        self.kpqp = self.KP * self.QP  #(k'q')
        self.dd   = self.D * self.D    #(dd)
        self.Pq   = self.P * self.Q    #(Pq)
        self.Pqp  = self.P * self.QP   #(Pq')
        self.qd   = self.Q * self.D    #(qd)
        self.qpd  = self.QP * self.D   #(q'd)

        # #Transverse vector products defined after eq.(241c) -----------------------
        self.kk_T   = self.TProduct(self.K,self.K)
        self.kkp_T  = self.kk_T
        self.kqp_T  = self.TProduct(self.K,self.QP)
        self.kd_T   = -1.* self.kqp_T
        self.dd_T   = self.TProduct(self.D,self.D)
        self.kpqp_T = self.TProduct(self.KP,self.QP)
        self.kP_T   = self.TProduct(self.K,self.P)
        self.kpP_T  = self.TProduct(self.KP,self.P)
        self.qpP_T  = self.TProduct(self.QP,self.P)
        self.kpd_T  = self.TProduct(self.KP,self.D)
        self.qpd_T  = self.TProduct(self.QP,self.D)

        #____________________________________________________________________________________

    def GetBHUUxs(self, phi, F1, F2) :

        self.Set4VectorsPhiDep(phi)
        self.Set4VectorProducts(phi)

        # Coefficients of the BH unpolarized structure function FUUBH
        self.AUUBH = ( (8. * self.M2) / (self.t * self.kqp * self.kpqp) ) * ( (4. * self.tau * (self.kP * self.kP + self.kpP * self.kpP) ) - ( (self.tau + 1.) * (self.kd * self.kd + self.kpd * self.kpd) ) )
        self.BUUBH = ( (16. * self.M2) / (self.t* self.kqp * self.kpqp) ) * (self.kd * self.kd + self.kpd * self.kpd)

        # Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
        # I multiply by 2 because I think Auu and Buu are missing a factor 2
        self.con_AUUBH = 2. * self.AUUBH * self.GeV2nb * self.jcob
        self.con_BUUBH = 2. * self.BUUBH * self.GeV2nb * self.jcob

        # Unpolarized Coefficients multiplied by the Form Factors
        self.bhAUU = (self.Gamma/self.t) * self.con_AUUBH * ( F1 * F1 + self.tau * F2 * F2 )
        self.bhBUU = (self.Gamma/self.t) * self.con_BUUBH * ( self.tau * ( F1 + F2 ) * ( F1 + F2 ) )

        # Unpolarized BH cross section
        self.xbhUU = self.bhAUU + self.bhBUU

        return self.xbhUU

        #____________________________________________________________________________________

    def GetIUUxs(self, phi, F1, F2, ReH, ReE, ReHtilde) :

        self.Set4VectorsPhiDep(phi)
        self.Set4VectorProducts(phi)

        # Interference coefficients given on eq. (241a,b,c)--------------------
        self.AUUI = -4.0 / (self.kqp * self.kpqp) * (( self.QQ + self.t ) * ( 2.0 * ( self.kP + self.kpP ) * self.kk_T + ( self.Pq * self.kqp_T ) + 2.* ( self.kpP * self.kqp ) - 2.* ( self.kP * self.kpqp ) + self.kpqp * self.kP_T + self.kqp * self.kpP_T - 2.*self.kkp * self.kP_T )
         + ( self.QQ - self.t + 4.* self.kd ) * ( self.Pqp * ( self.kkp_T + self.kqp_T - 2.* self.kkp ) + 2.* self.kkp * self.qpP_T - self.kpqp * self.kP_T - self.kqp * self.kpP_T ) )

        self.BUUI = 2.0 * self.xi / ( self.kqp * self.kpqp) * ( ( self.QQ + self.t ) * ( 2.* self.kk_T * ( self.kd + self.kpd ) + self.kqp_T * ( self.qd - self.kqp - self.kpqp + 2.*self.kkp ) + 2.* self.kqp * self.kpd - 2.* self.kpqp * self.kd ) +
                                                    ( self.QQ - self.t + 4.* self.kd ) * ( ( self.kk_T - 2.* self.kkp ) * self.qpd - self.kkp * self.dd_T - 2.* self.kd_T * self.kqp ) )
        self.CUUI = 2.0 / ( self.kqp * self.kpqp) * ( -1. * ( self.QQ + self.t ) * ( 2.* self.kkp - self.kpqp - self.kqp + 2.* self.xi * (2.* self.kkp * self.kP_T - self.kpqp * self.kP_T - self.kqp * self.kpP_T) ) * self.kd_T +
                                                  ( self.QQ - self.t + 4.* self.kd ) * ( ( self.kqp + self.kpqp ) * self.kd_T + self.dd_T * self.kkp + 2.* self.xi * ( self.kkp * self.qpP_T - self.kpqp * self.kP_T - self.kqp * self.kpP_T ) ) )

        # Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian

        #print(self.AUUI, self.GeV2nb, self.jcob)
        self.con_AUUI = self.AUUI * self.GeV2nb * self.jcob
        self.con_BUUI = self.BUUI * self.GeV2nb * self.jcob
        self.con_CUUI = self.CUUI * self.GeV2nb * self.jcob

        #Unpolarized Coefficients multiplied by the Form Factors

        self.iAUU = (self.Gamma/(-self.t * self.QQ)) * tf.cos( phi * self.RAD ) * self.con_AUUI * ( F1 * ReH + self.tau * F2 * ReE )
        self.iBUU = (self.Gamma/(-self.t * self.QQ)) * tf.cos( phi * self.RAD ) * self.con_BUUI * ( F1 + F2 ) * ( ReH + ReE )
        self.iCUU = (self.Gamma/(-self.t * self.QQ)) * tf.cos( phi * self.RAD ) * self.con_CUUI * ( F1 + F2 ) * ReHtilde


        # Unpolarized BH-DVCS interference cross section
        self.xIUU = self.iAUU + self.iBUU + self.iCUU

        return self.xIUU

    # I don't think this function is necessary
    # def TotalUUXS(self, phi, k, QQ, xB, t, F1, F2, ReH, ReE, ReHtilde, dvcs):
	#     # Set QQ, xB, t and k and calculate 4-vector products
    #     #print(par)
    #     self.SetKinematics(QQ, xB, t, k)
    #     self.Set4VectorsPhiDep(phi)
    #     self.Set4VectorProducts(phi)
    #
    #     xsbhuu	 = self.GetBHUUxs(phi, F1, F2)
    #     xsiuu	 = self.GetIUUxs(phi, F1, F2, ReH, ReE, ReHtilde)
    #     tot_sigma_uu = xsbhuu + xsiuu + dvcs # Constant added to account for DVCS contribution
    #     #print(xsbhuu, " ", xsiuu, " ", tot_sigma_uu)
    #     return tot_sigma_uu

    def TotalUUXS(self, x, ReH, ReE, ReHtilde): # originally named TotalUUXS_curve_fit, but original TotalUUXS is unnecessary
        """
        params:
            x: should be a 2d numpy array (or tf) with columns in the
             order of phi, k, QQ, xb, t, F1, F2, dvcs
            ReH: scalar or 1d of length len(x)
            ReE: same as ReH
            ReHtilde: same as ReH
        returns:
            tf.float64 array of length len(x)
        """

        phi, k, QQ, xb, t, F1, F2, dvcs = x # a little weird but I like the readability


	    # Set QQ, xB, t and k and calculate 4-vector products
        self.SetKinematics(QQ, xb, t, k)
        self.Set4VectorsPhiDep(phi)
        self.Set4VectorProducts(phi)

        xsbhuu	 = self.GetBHUUxs(phi, F1, F2)
        xsiuu	 = self.GetIUUxs(phi, F1, F2, ReH, ReE, ReHtilde)


        tot_sigma_uu = xsbhuu + xsiuu +  dvcs # Constant added to account for DVCS contribution
        return tot_sigma_uu
