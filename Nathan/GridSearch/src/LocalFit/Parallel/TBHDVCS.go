package main

import (
	"fmt"
	"math"
	"os"
)

/*
const ALP_INV float64 = 137.0359998 // 1 / Electromagnetic Fine Structure Constant
const PI float64 = 3.1415926535
const RAD float64 = (PI / 180.)
const M float64 = 0.938272    //Mass of the proton in GeV
const GeV2nb float64 = 389379 // Conversion from GeV to NanoBarn = .389379*1000000
*/
type BHDVCS struct {
	QQ, t, Gamma, tau, F1, F2    float64     // kinematic variables
	con_AUUI, con_BUUI, con_CUUI []float64   // intermediate getIUU computations
	xbhUU                        []float64   // getBHUU results
	phiValues                    map[int]int // values of phi
}

func (b *BHDVCS) getPhiIndex(phi int) int {
	index, ok := b.phiValues[phi]

	if !ok {
		fmt.Println("Error: phi value not found")
		os.Exit(1)
	}

	return index
}

func (b *BHDVCS) init(QQ, x, t, k, F1, F2 float64, phiValuesArr []int) {
	lengthOfPhiValues := len(phiValuesArr)
	b.phiValues = make(map[int]int)

	for index, phi := range phiValuesArr {
		b.phiValues[phi] = index
	}

	b.t = t
	b.QQ = QQ
	b.F1 = F1
	b.F2 = F2

	// Set Kinematics
	M2 := M * M //Mass of the proton  squared in GeV
	//fractional energy of virtual photon
	y := QQ / (2. * M * k * x) // From eq. (23) where gamma is substituted from eq (12c)
	//squared gamma variable ratio of virtuality to energy of virtual photon
	gg := 4. * M2 * x * x / QQ // This is gamma^2 [from eq. (12c)]
	//ratio of longitudinal to transverse virtual photon flux
	//e := ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ) // epsilon eq. (32)
	//Skewness parameter
	xi := 1. * x * ((1. + t/(2.*QQ)) / (2. - x + x*t/QQ)) // skewness parameter eq. (12b) dnote: there is a minus sign on the write up that shouldn't be there
	//Minimum t value
	//tmin := ( QQ * ( 1. - math.Sqrt( 1. + gg ) + gg / 2. ) ) / ( x * ( 1. - math.Sqrt( 1. + gg ) + gg / ( 2.* x ) ) ) // minimum t eq. (29)
	//Final Lepton energy
	kpr := k * (1. - y) // k' from eq. (23)
	//outgoing photon energy
	qp := t/2./M + k - kpr //q' from eq. bellow to eq. (25) that has no numbering. Here nu := k - k' := k * y
	//Final proton Energy
	//po := M - t / 2. / M // This is p'_0 from eq. (28b)
	//pmag := math.Sqrt( ( -1* t ) * ( 1. - (t / (4. * M * M ))) ) // p' magnitude from eq. (28b)
	//Angular Kinematics of outgoing photon
	cth := -1. / math.Sqrt(1.+gg) * (1. + gg/2.*(1.+t/QQ)/(1.+x*t/QQ)) // This is math.Cos(theta) eq. (26)
	theta := math.Acos(cth)                                            // theta angle
	//print('Theta: ', theta)
	//Lepton Angle Kinematics of initial lepton
	sthl := math.Sqrt(gg) / math.Sqrt(1.+gg) * (math.Sqrt(1. - y - y*y*gg/4.)) // math.Sin(theta_l) from eq. (22a)
	cthl := -1. / math.Sqrt(1.+gg) * (1. + y*gg/2.)                            // math.Cos(theta_l) from eq. (22a)
	//ratio of momentum transfer to proton mass
	b.tau = -0.25 * t / M2

	// phi independent 4 - momenta vectors defined on eq. (21) -------------
	K := TLorentzVector{k * sthl, 0.0, k * cthl, k}
	KP := TLorentzVector{K.x, 0.0, k * (cthl + y*math.Sqrt(1.+gg)), kpr}
	Q := K.minus(KP)
	p := TLorentzVector{0.0, 0.0, 0.0, M}

	// Sets the Mandelstam variable s which is the center of mass energy
	s := p.plus(K).multiply(p.plus(K))

	// The Gamma factor in front of the cross section
	b.Gamma = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / (s - M2) / (s - M2) / math.Sqrt(1.+gg) / x

	// Defurne's Jacobian
	jcob := 1. / (2. * M * x * K.t) * 2. * PI * 2.

	// Set 4 Vector Products, phi-independent

	// 4-vectors products (phi - independent)
	kkp := K.multiply(KP) //(kk')
	//kq   = TLorentzVector_mul(K, Q)    //(kq)
	//kp   = TLorentzVector_mul(K, p)    //(pk)
	//kpp  = TLorentzVector_mul(KP, p)   //(pk')

	b.xbhUU = make([]float64, lengthOfPhiValues)
	b.con_AUUI = make([]float64, lengthOfPhiValues)
	b.con_BUUI = make([]float64, lengthOfPhiValues)
	b.con_CUUI = make([]float64, lengthOfPhiValues)

	for i := 0; i < lengthOfPhiValues; i++ {
		phiValue := float64(phiValuesArr[i])
		// Set4VectorsPhiDep

		QP := TLorentzVector{qp * math.Sin(theta) * math.Cos(phiValue*RAD), qp * math.Sin(theta) * math.Sin(phiValue*RAD), qp * math.Cos(theta), qp}
		D := Q.minus(QP) // delta vector eq. (12a)
		//print(D, "\n", Q, "\n", QP)
		pp := p.plus(D) // p' from eq. (21)
		P := p.plus(pp)
		P = TLorentzVector{.5 * P.x, .5 * P.y, .5 * P.z, .5 * P.t}

		// Set 4 Vector Products, phi-dependent

		// 4-vectors products (phi - dependent)
		kd := K.multiply(D)     //(kd)
		kpd := KP.multiply(D)   //(k'd)
		kP := K.multiply(P)     //(kP)
		kpP := KP.multiply(P)   //(k'P)
		kqp := K.multiply(QP)   //(kq')
		kpqp := KP.multiply(QP) //(k'q')
		//dd   := D.multiply(D)    //(dd)
		Pq := P.multiply(Q)   //(Pq)
		Pqp := P.multiply(QP) //(Pq')
		qd := Q.multiply(D)   //(qd)
		qpd := QP.multiply(D) //(q'd)

		// //Transverse vector products defined after eq.(241c) -----------------------
		kk_T := BHDVCS_TProduct(K, K)
		kkp_T := kk_T
		kqp_T := BHDVCS_TProduct(K, QP)
		kd_T := -1. * kqp_T
		dd_T := BHDVCS_TProduct(D, D)
		//kpqp_T := BHDVCS_TProduct(KP, QP)
		kP_T := BHDVCS_TProduct(K, P)
		kpP_T := BHDVCS_TProduct(KP, P)
		qpP_T := BHDVCS_TProduct(QP, P)
		//kpd_T  := BHDVCS_TProduct(KP, D)
		//qpd_T  := BHDVCS_TProduct(QP, D)

		// Get BHUUxs

		// Coefficients of the BH unpolarized structure function FUUBH
		AUUBH := ((8. * M2) / (t * kqp * kpqp)) * ((4. * b.tau * (kP*kP + kpP*kpP)) - ((b.tau + 1.) * (kd*kd + kpd*kpd)))
		BUUBH := ((16. * M2) / (t * kqp * kpqp)) * (kd*kd + kpd*kpd)

		// Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
		// I multiply by 2 because I think Auu and Buu are missing a factor 2
		con_AUUBH := 2. * AUUBH * GeV2nb * jcob
		con_BUUBH := 2. * BUUBH * GeV2nb * jcob

		// Unpolarized Coefficients multiplied by the Form Factors
		bhAUU := (b.Gamma / t) * con_AUUBH * (F1*F1 + b.tau*F2*F2)
		bhBUU := (b.Gamma / t) * con_BUUBH * (b.tau * (F1 + F2) * (F1 + F2))

		// Unpolarized BH cross section
		b.xbhUU[i] = bhAUU + bhBUU

		// first half of Get IUUxs

		// Interference coefficients given on eq. (241a,b,c)--------------------
		AUUI := -4.0 / (kqp * kpqp) * ((QQ+t)*(2.0*(kP+kpP)*kk_T+(Pq*kqp_T)+2.*(kpP*kqp)-2.*(kP*kpqp)+kpqp*kP_T+kqp*kpP_T-2.*kkp*kP_T) + (QQ-t+4.*kd)*(Pqp*(kkp_T+kqp_T-2.*kkp)+2.*kkp*qpP_T-kpqp*kP_T-kqp*kpP_T))

		BUUI := 2.0 * xi / (kqp * kpqp) * ((QQ+t)*(2.*kk_T*(kd+kpd)+kqp_T*(qd-kqp-kpqp+2.*kkp)+2.*kqp*kpd-2.*kpqp*kd) + (QQ-t+4.*kd)*((kk_T-2.*kkp)*qpd-kkp*dd_T-2.*kd_T*kqp))
		CUUI := 2.0 / (kqp * kpqp) * (-1.*(QQ+t)*(2.*kkp-kpqp-kqp+2.*xi*(2.*kkp*kP_T-kpqp*kP_T-kqp*kpP_T))*kd_T + (QQ-t+4.*kd)*((kqp+kpqp)*kd_T+dd_T*kkp+2.*xi*(kkp*qpP_T-kpqp*kP_T-kqp*kpP_T)))

		// Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian

		//print(AUUI, GeV2nb, jcob)
		b.con_AUUI[i] = AUUI * GeV2nb * jcob
		b.con_BUUI[i] = BUUI * GeV2nb * jcob
		b.con_CUUI[i] = CUUI * GeV2nb * jcob
	}

}

func BHDVCS_TProduct(v1, v2 TLorentzVector) float64 {
	return (v1.Px() * v2.Px()) + (v1.Py() * v2.Py())
}

func (b *BHDVCS) BHDVCS_GetIUUxs(phi int, ReH float64, ReE float64, ReHtilde float64) float64 {
	i := b.getPhiIndex(phi)
	phiFloat64 := float64(phi)

	//Unpolarized Coefficients multiplied by the Form Factors
	iAUU := (b.Gamma / (-b.t * b.QQ)) * math.Cos(phiFloat64*RAD) * b.con_AUUI[i] * (b.F1*ReH + b.tau*b.F2*ReE)
	iBUU := (b.Gamma / (-b.t * b.QQ)) * math.Cos(phiFloat64*RAD) * b.con_BUUI[i] * (b.F1 + b.F2) * (ReH + ReE)
	iCUU := (b.Gamma / (-b.t * b.QQ)) * math.Cos(phiFloat64*RAD) * b.con_CUUI[i] * (b.F1 + b.F2) * ReHtilde

	// Unpolarized BH-DVCS interference cross section
	return iAUU + iBUU + iCUU // return xIUU
}

func (b *BHDVCS) getBHUU_plus_getIUU(phi int, ReH, ReE, ReHtilde float64) float64 {
	i := b.getPhiIndex(phi)

	return b.xbhUU[i] + b.BHDVCS_GetIUUxs(phi, ReH, ReE, ReHtilde)
}
