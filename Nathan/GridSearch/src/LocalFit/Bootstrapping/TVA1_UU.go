package main

import (
	"fmt"
	"math"
	"os"
)

const ALP_INV float64 = 137.0359998 // 1 / Electromagnetic Fine Structure Constant
const PI float64 = 3.1415926535
const RAD float64 = (PI / 180.)
const M float64 = 0.938272    //Mass of the proton in GeV
const GeV2nb float64 = 389379 // Conversion from GeV to NanoBarn = .389379*1000000

type TVA1_UU struct {
	QQ, t, Gamma, tau, F1, F2    float64   // kinematic variables
	con_AUUI, con_BUUI, con_CUUI []float64 // intermediate getIUU computations
	xbhUU                        []float64 // getBHUU results
	phiValues                    []float64 // values of phi
}

func (self *TVA1_UU) init(QQ, x, t, k, F1, F2 float64, phiValues []float64) {
	var M2 float64 = M * M
	var jcob float64 = 2 * PI
	var y float64 = QQ / (2 * M * k * x)
	var gg float64 = 4 * M2 * x * x / QQ
	var xi float64 = x * (1 + t/(2*QQ)) / (2 - x + (x * t / QQ))
	var kpr float64 = k * (1 - y)
	var qp float64 = ((t / 2) / M) + k - kpr
	var cth float64 = -1 / math.Sqrt(1+gg) * (1 + ((gg / 2) * (1 + (t / QQ)) / (1 + (x * t / QQ))))
	var theta float64 = math.Acos(cth)
	var sthl float64 = math.Sqrt(gg) / math.Sqrt(1+gg) * (math.Sqrt(1 - y - (y * y * gg / 4)))
	var cthl float64 = -1 / math.Sqrt(1+gg) * (1 + (y * gg / 2))

	var K TLorentzVector = TLorentzVector{k * sthl, 0.0, k * cthl, k}
	var KP TLorentzVector = TLorentzVector{K.parenthesesOperator(0), 0.0, k * (cthl + (y * math.Sqrt(1+gg))), kpr}
	var Q TLorentzVector = K.minus(KP)
	var p TLorentzVector = TLorentzVector{0.0, 0.0, 0.0, M}

	var s float64 = p.plus(K).multiply(p.plus(K))
	var kkp float64 = K.multiply(KP)
	var kk_T float64 = TVA1_UU_TProduct(K, K)
	var kkp_T float64 = kk_T

	self.F1 = F1
	self.F2 = F2
	self.QQ = QQ
	self.t = t
	self.tau = -0.25 * t / M2
	self.Gamma = 1 / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16 / (s - M2) / (s - M2) / math.Sqrt(1+gg) / x

	self.phiValues = make([]float64, len(phiValues))
	self.con_AUUI = make([]float64, len(phiValues))
	self.con_BUUI = make([]float64, len(phiValues))
	self.con_CUUI = make([]float64, len(phiValues))
	self.xbhUU = make([]float64, len(phiValues))

	for i := 0; i < len(phiValues); i++ {
		// Set4VectorsPhiDep
		var QP TLorentzVector = TLorentzVector{qp * math.Sin(theta) * math.Cos(phiValues[i]*RAD), qp * math.Sin(theta) * math.Sin(phiValues[i]*RAD), qp * math.Cos(theta), qp}
		var D TLorentzVector = Q.minus(QP)
		var pp TLorentzVector = p.plus(D)
		var P TLorentzVector = TLorentzVector{(p.x + pp.x) / 2, (p.y + pp.y) / 2, (p.z + pp.z) / 2, (p.t + pp.t) / 2}

		// Set4VectorProducts
		var kd float64 = K.multiply(D)
		var kpd float64 = KP.multiply(D)
		var kP float64 = K.multiply(P)
		var kpP float64 = KP.multiply(P)
		var kqp float64 = K.multiply(QP)
		var kpqp float64 = KP.multiply(QP)
		var Pqp float64 = P.multiply(QP)
		var qpd float64 = QP.multiply(D)

		var kqp_T float64 = TVA1_UU_TProduct(K, QP)
		var kd_T float64 = -1 * kqp_T
		var dd_T float64 = TVA1_UU_TProduct(D, D)
		var kpqp_T float64 = kqp_T
		var kP_T float64 = TVA1_UU_TProduct(K, P)
		var kpP_T float64 = TVA1_UU_TProduct(KP, P)
		var qpP_T float64 = TVA1_UU_TProduct(QP, P)
		var kpd_T float64 = -1 * kqp_T
		var qpd_T float64 = -1 * dd_T

		var Dplus float64 = (0.5 / kpqp) - (0.5 / kqp)
		var Dminus float64 = (-0.5 / kpqp) - (0.5 / kqp)

		// getBHUU
		var AUUBH float64 = (8. * M2) / (self.t * kqp * kpqp) * ((4. * self.tau * (kP*kP + kpP*kpP)) - ((self.tau + 1.) * (kd*kd + kpd*kpd)))
		var BUUBH float64 = (16. * M2) / (self.t * kqp * kpqp) * (kd*kd + kpd*kpd)

		var con_AUUBH float64 = AUUBH * GeV2nb * jcob
		var con_BUUBH float64 = BUUBH * GeV2nb * jcob

		var bhAUU float64 = (self.Gamma / self.t) * con_AUUBH * (F1*F1 + self.tau*F2*F2)
		var bhBUU float64 = (self.Gamma / self.t) * con_BUUBH * (self.tau * (F1 + F2) * (F1 + F2))

		// first half of getIUU
		var AUUI float64 = -4. * math.Cos(phiValues[i]*RAD) * (Dplus*((kqp_T-2.*kk_T-2.*kqp)*kpP+(2.*kpqp-2.*kkp_T-kpqp_T)*kP+kpqp*kP_T+kqp*kpP_T-2.*kkp*kP_T) - Dminus*((2.*kkp-kpqp_T-kkp_T)*Pqp+2.*kkp*qpP_T-kpqp*kP_T-kqp*kpP_T))
		var BUUI float64 = -2. * xi * math.Cos(phiValues[i]*RAD) * (Dplus*((kqp_T-2.*kk_T-2.*kqp)*kpd+(2.*kpqp-2.*kkp_T-kpqp_T)*kd+kpqp*kd_T+kqp*kpd_T-2.*kkp*kd_T) - Dminus*((2.*kkp-kpqp_T-kkp_T)*qpd+2.*kkp*qpd_T-kpqp*kd_T-kqp*kpd_T))
		var CUUI float64 = -2. * math.Cos(phiValues[i]*RAD) * (Dplus*(2.*kkp*kd_T-kpqp*kd_T-kqp*kpd_T+4.*xi*kkp*kP_T-2.*xi*kpqp*kP_T-2.*xi*kqp*kpP_T) - Dminus*(kkp*qpd_T-kpqp*kd_T-kqp*kpd_T+2.*xi*kkp*qpP_T-2.*xi*kpqp*kP_T-2.*xi*kqp*kpP_T))

		self.phiValues[i] = phiValues[i]
		self.xbhUU[i] = bhAUU + bhBUU
		self.con_AUUI[i] = AUUI * GeV2nb * jcob
		self.con_BUUI[i] = BUUI * GeV2nb * jcob
		self.con_CUUI[i] = CUUI * GeV2nb * jcob
	}
}

func (self *TVA1_UU) getPhiIndex(phi float64) int {
	for i := 0; i < len(self.phiValues); i++ {
		if math.Abs(self.phiValues[i]-phi) < 0.000001 {
			return i
		}
	}

	fmt.Errorf("Error: phi value not found\n")
	os.Exit(255)
	return -1 // mandatory in Go even though it will never reach here
}

func TVA1_UU_TProduct(v1, v2 TLorentzVector) float64 {
	return (v1.Px() * v2.Px()) + (v1.Py() * v2.Py())
}

func (self *TVA1_UU) getBHUU_plus_getIUU(phi, ReH, ReE, ReHtilde float64) float64 {
	var phiIndex int = self.getPhiIndex(phi)

	// second half of getIUU
	var something float64 = (self.Gamma / (math.Abs(self.t) * self.QQ))

	var iAUU float64 = something * self.con_AUUI[phiIndex] * ((self.F1 * ReH) + (self.tau * self.F2 * ReE))
	var iBUU float64 = something * self.con_BUUI[phiIndex] * (self.F1 + self.F2) * (ReH + ReE)
	var iCUU float64 = something * self.con_CUUI[phiIndex] * (self.F1 + self.F2) * ReHtilde

	var xIUU float64 = iAUU + iBUU + iCUU

	// getBHUU + getIUU = xbhUU + (-1 * xIUU)
	return self.xbhUU[phiIndex] - xIUU
}
