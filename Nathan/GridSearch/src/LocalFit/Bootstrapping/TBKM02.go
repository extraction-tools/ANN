package main

/*
import (
	"math"
	"math/cmplx"
)

type TBKM02 struct {
	QQ, t, Gamma, tau, F1, F2    float64     // kinematic variables
	con_AUUI, con_BUUI, con_CUUI []float64   // intermediate getIUU computations
	bhUU                         []float64   // getBHUU results
	phiValues                    map[int]int // values of phi
}

func (tbk *TBKM02) init(QQ, x, t, k, F1, F2 float64, phiValuesArr []float64) {
	lengthOfPhiValues := len(phiValuesArr)
	tbk.phiValues = make(map[int]int)

	M2 := M * M
	ee := 4. * M2 * x * x / QQ               // epsilon squared
	y := math.Sqrt(QQ) / (math.Sqrt(ee) * k) // lepton energy fraction
	//xi := x * (1. + t/2./QQ) / (2. - x + x*t/QQ)                                             // Generalized Bjorken variable
	Gamma1 := x * y * y / ALP_INV / ALP_INV / ALP_INV / PI / 8. / QQ / QQ / math.Sqrt(1.+ee) // factor in front of the cross section, eq. (22)
	//s := 2.*M*k + M2
	tmin := -QQ * (2.*(1.-x)*(1.-math.Sqrt(1.+ee)) + ee) / (4.*x*(1.-x) + ee)                                                             // eq. (31)
	K2 := -(t / QQ) * (1. - x) * (1. - y - y*y*ee/4.) * (1. - tmin/t) * (math.Sqrt(1.+ee) + ((4.*x*(1.-x)+ee)/(4.*(1.-x)))*((t-tmin)/QQ)) // eq. (30)

	// first part of BHUU
	// BH unpolarized Fourier harmonics eqs. (35 - 37)
	c0_BH := 8.*K2*((2.+3.*ee)*(QQ/t)*(F1*F1-F2*F2*t/(4.*M2))+2.*x*x*(F1+F2)*(F1+F2)) +
		(2.-y)*(2.-y)*((2.+ee)*((4.*x*x*M2/t)*(1.+t/QQ)*(1.+t/QQ)+4.*(1.-x)*(1.+x*t/QQ))*(F1*F1-F2*F2*t/(4.*M2))+
			4.*x*x*(x+(1.-x+ee/2.)*(1.-t/QQ)*(1.-t/QQ)-x*(1.-2.*x)*t*t/(QQ*QQ))*(F1+F2)*(F1+F2)) +
		8.*(1.+ee)*(1.-y-ee*y*y/4.)*(2.*ee*(1.-t/(4.*M2))*(F1*F1-F2*F2*t/(4.*M2))-x*x*(1.-t/QQ)*(1.-t/QQ)*(F1+F2)*(F1+F2))

	c1_BH := 8. * math.Sqrt(K2) * (2. - y) * ((4.*x*x*M2/t-2.*x-ee)*(F1*F1-F2*F2*t/(4.*M2)) + 2.*x*x*(1.-(1.-2.*x)*t/QQ)*(F1+F2)*(F1+F2))

	c2_BH := 8. * x * x * K2 * ((4.*M2/t)*(F1*F1-F2*F2*t/(4.*M2)) + 2.*(F1+F2)*(F1+F2))

	tbk.bhUU = make([]float64, lengthOfPhiValues)

	for i, phi := range phiValuesArr {
		tbk.phiValues[int(phi)] = i

		// K*D 4-vector product (phi-dependent)
		KD := -QQ / (2. * y * (1. + ee)) * (1. + 2.*math.Sqrt(K2)*math.Cos(PI-(phi*RAD)) - t/QQ*(1.-x*(2.-y)+y*ee/2.) + y*ee/2.) // eq. (29)

		// lepton BH propagators P1 and P2 (contaminating phi-dependence)
		P1 := 1. + 2.*KD/QQ
		P2 := t/QQ - 2.*KD/QQ

		// second part of bhuu
		// BH squared amplitude eq (25) divided by e^6
		Amp2_BH := 1. / (x * x * y * y * (1. + ee) * (1. + ee) * t * P1 * P2) * (c0_BH + c1_BH*math.Cos(PI-(phi*RAD)) + c2_BH*math.Cos(2.*(PI-(phi*RAD))))

		Amp2_BH = GeV2nb * Amp2_BH // convertion to nb

		dsigma_BH := Gamma1 * Amp2_BH

		tbk.bhuu[i] = dsigma_BH
	}

}

// complex and complex conjugate numbers product
// ( C D* ) product
func cdstar(c, d complex128) complex128 {
	dstar := cmplx.Conj(d)

	return (real(c)*real(dstar) - imag(c)*imag(dstar)) + (real(c)*imag(dstar)+imag(c)*real(dstar))*complex(0, 1)
}

func Rho2(c complex128) float64 {
	return real(c)*real(c) + imag(c)*imag(c)
}

// Pure DVCS Unpolarized Cross Section with just considering thre c0 term
func (t *TBKM02) DVCSUU(t2cffs [4]complex128) float64 {
	// t2cffs = { H, E , Htilde, Etilde } Twist-2 Compton Form Factors
	H := t2cffs[0]
	E := t2cffs[1]
	Htilde := t2cffs[2]
	Etilde := t2cffs[3]

	// c coefficients (eq. 66) for pure DVCS .
	c_dvcs := 1. / (2. - x) / (2. - x) * (complex(4.*(1-x)*(Rho2(H)+Rho2(Htilde)), 0) - complex(x, 0)*complex(x, 0)*(cdstar(H, E)+cdstar(E, H)+cdstar(Htilde, Etilde)+cdstar(Etilde, Htilde)) -
		complex((x*x+(2.-x)*(2.-x)*t/4./M2)*E.Rho2(), 0) - complex((x*x*t/4./M2)*Rho2(Etilde), 0))

	// Pure DVCS unpolarized Fourier harmonics eqs. (43)
	c0_dvcs := 2. * (2. - 2.*y + y*y) * c_dvcs

	// DVCS squared amplitude eq (26) divided by e^6
	Amp2_DVCS := 1. / (y * y * QQ) * c0_dvcs

	Amp2_DVCS = GeV2nb * Amp2_DVCS // convertion to nb

	dsigma_DVCS := Gamma1 * Amp2_DVCS

	return dsigma_DVCS
}

func (t *TBKM02) IUU(phi int, t2cffs [4]complex128) float64 { // Interference Unpolarized Cross Section (writting it as in Liuti's comparison paper, i.e just c0 and c1 terms)

	// t2cffs_I = { H, E , Htilde, Etilde } Twist-2 Compton Form Factors present in the interference term
	H := t2cffs[0]
	E := t2cffs[1]
	Htilde := t2cffs[2]
	Etilde := t2cffs[3] // This CFF does not appear in the interference

	//Coefficients in from to the CFFs
	A := -8.*K2*(2.-y)*(2.-y)*(2.-y)/(1.-y) - 8.*(2.-y)*(1.-y)*(2.-x)*t/QQ - 8.*sqrt(K2)*(2.-2.*y+y*y)*cos(PI-(phi*RAD))
	B := 8. * x * x * (2. - y) * (1 - y) / (2. - x) * t / QQ
	C := x / (2. - x) * (-8.*K2*(2.-y)*(2.-y)*(2.-y)/(1.-y) - 8.*sqrt(K2)*(2.-2.*y+y*y)*cos(PI-(phi*RAD)))

	// BH-DVCS interference squared amplitude eq (27) divided by e^6
	Amp2_I := 1. / (x * y * y * y * t * P1 * P2) * (A*(F1*H.Re()-t/4./M2*F2*E.Re()) + B*(F1+F2)*(H.Re()+E.Re()) + C*(F1+F2)*Htilde.Re())

	Amp2_I := GeV2nb * Amp2_I // convertion to nb

	dsigma_I := Gamma1 * Amp2_I

	return dsigma_I
}
*/
