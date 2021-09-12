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
	QQ, t                float64
	xi, M2, tau, e, tmin float64
	Gamma                float64
	jcob                 float64

	QP, D, P []TLorentzVector

	kkp                                               float64 // not dependent on phi
	kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd []float64

	kk_T, kkp_T                                                 float64 // not dependent on phi
	kqp_T, kd_T, dd_T, kpqp_T, kP_T, kpP_T, qpP_T, kpd_T, qpd_T []float64

	Dplus, Dminus []float64

	phiValues map[int]int
}

func (t *TVA1_UU) init(_QQ, x, _t, k float64, phiValuesArr []int) {
	lengthOfPhiValues := len(phiValuesArr)
	t.phiValues = make(map[int]int)

	for index, phi := range phiValuesArr {
		t.phiValues[phi] = index
	}

	t.QQ = _QQ
	t.t = _t
	t.M2 = M * M
	t.jcob = 2 * PI

	y := _QQ / (2 * M * k * x)
	gg := 4 * t.M2 * x * x / _QQ
	t.e = (1 - y - (y * y * (gg / 4))) / (1 - y + (y * y / 2) + (y * y * (gg / 4)))
	t.xi = x * (1 + _t/(2*_QQ)) / (2 - x + (x * _t / _QQ))
	t.tmin = -1 * _QQ * (1 - math.Sqrt(1+gg) + (gg / 2)) / (x * (1 - math.Sqrt(1+gg) + (gg / (2 * x))))
	kpr := k * (1 - y)
	qp := ((_t / 2) / M) + k - kpr
	cth := -1 / math.Sqrt(1+gg) * (1 + ((gg / 2) * (1 + (_t / _QQ)) / (1 + (x * _t / _QQ))))
	theta := math.Acos(cth)
	sthl := math.Sqrt(gg) / math.Sqrt(1+gg) * (math.Sqrt(1 - y - (y * y * gg / 4)))
	cthl := -1 / math.Sqrt(1+gg) * (1 + (y * gg / 2))
	t.tau = -0.25 * _t / t.M2

	K := TLorentzVector{
		k * sthl,
		0.0,
		k * cthl,
		k,
	}

	KP := TLorentzVector{
		K.parenthesesOperator(0),
		0.0,
		k * (cthl + (y * math.Sqrt(1+gg))),
		kpr,
	}

	Q := K.minus(KP)
	p := TLorentzVector{
		0.0,
		0.0,
		0.0,
		M,
	}

	s := p.plus(K).multiply(p.plus(K))
	t.Gamma = (((((((((1 / ALP_INV) / ALP_INV) / ALP_INV) / PI) / PI) / 16) / (s - t.M2)) / (s - t.M2)) / math.Sqrt(1+gg)) / x

	t.kkp = K.multiply(KP)

	t.kk_T = TVA1_UU_TProduct(K, K)
	t.kkp_T = t.kk_T

	t.QP = make([]TLorentzVector, lengthOfPhiValues)
	t.D = make([]TLorentzVector, lengthOfPhiValues)
	t.P = make([]TLorentzVector, lengthOfPhiValues)

	t.kd = make([]float64, lengthOfPhiValues)
	t.kpd = make([]float64, lengthOfPhiValues)
	t.kP = make([]float64, lengthOfPhiValues)
	t.kpP = make([]float64, lengthOfPhiValues)
	t.kqp = make([]float64, lengthOfPhiValues)
	t.kpqp = make([]float64, lengthOfPhiValues)
	t.dd = make([]float64, lengthOfPhiValues)
	t.Pq = make([]float64, lengthOfPhiValues)
	t.Pqp = make([]float64, lengthOfPhiValues)
	t.qd = make([]float64, lengthOfPhiValues)
	t.qpd = make([]float64, lengthOfPhiValues)

	t.kqp_T = make([]float64, lengthOfPhiValues)
	t.kd_T = make([]float64, lengthOfPhiValues)
	t.dd_T = make([]float64, lengthOfPhiValues)
	t.kpqp_T = make([]float64, lengthOfPhiValues)
	t.kP_T = make([]float64, lengthOfPhiValues)
	t.kpP_T = make([]float64, lengthOfPhiValues)
	t.qpP_T = make([]float64, lengthOfPhiValues)
	t.kpd_T = make([]float64, lengthOfPhiValues)
	t.qpd_T = make([]float64, lengthOfPhiValues)

	t.Dplus = make([]float64, lengthOfPhiValues)
	t.Dminus = make([]float64, lengthOfPhiValues)

	for i := 0; i < lengthOfPhiValues; i++ {
		// Set4VectorsPhiDep
		t.QP[i] = TLorentzVector{
			qp * math.Sin(theta) * math.Cos(float64(phiValuesArr[i])*RAD),
			qp * math.Sin(theta) * math.Sin(float64(phiValuesArr[i])*RAD),
			qp * math.Cos(theta),
			qp,
		}
		t.D[i] = Q.minus(t.QP[i])
		pp := p.plus(t.D[i])
		t.P[i] = p.plus(pp)
		t.P[i] = TLorentzVector{
			0.5 * t.P[i].Px(),
			0.5 * t.P[i].Py(),
			0.5 * t.P[i].Pz(),
			0.5 * t.P[i].E(),
		}

		// Set4VectorProducts
		t.kd[i] = K.multiply(t.D[i])
		t.kpd[i] = KP.multiply(t.D[i])
		t.kP[i] = K.multiply(t.P[i])
		t.kpP[i] = KP.multiply(t.P[i])
		t.kqp[i] = K.multiply(t.QP[i])
		t.kpqp[i] = KP.multiply(t.QP[i])
		t.dd[i] = t.D[i].multiply(t.D[i])
		t.Pq[i] = t.P[i].multiply(Q)
		t.Pqp[i] = t.P[i].multiply(t.QP[i])
		t.qd[i] = Q.multiply(t.D[i])
		t.qpd[i] = t.QP[i].multiply(t.D[i])

		t.kqp_T[i] = TVA1_UU_TProduct(K, t.QP[i])
		t.kd_T[i] = -1 * t.kqp_T[i]
		t.dd_T[i] = TVA1_UU_TProduct(t.D[i], t.D[i])
		t.kpqp_T[i] = t.kqp_T[i]
		t.kP_T[i] = TVA1_UU_TProduct(K, t.P[i])
		t.kpP_T[i] = TVA1_UU_TProduct(KP, t.P[i])
		t.qpP_T[i] = TVA1_UU_TProduct(t.QP[i], t.P[i])
		t.kpd_T[i] = -1 * t.kqp_T[i]
		t.qpd_T[i] = -1 * t.dd_T[i]

		t.Dplus[i] = 0.5/t.kpqp[i] - 0.5/t.kqp[i]
		t.Dminus[i] = -0.5/t.kpqp[i] - 0.5/t.kqp[i]
	}
}

func (t *TVA1_UU) getPhiIndex(phi int) int {
	index, ok := t.phiValues[phi]

	if !ok {
		fmt.Println("Error: phi value not found")
		os.Exit(1)
	}

	return index
}

func TVA1_UU_TProduct(v1, v2 TLorentzVector) float64 {
	return (v1.Px() * v2.Px()) + (v1.Py() * v2.Py())
}

func (t *TVA1_UU) getBHUU_plus_getIUU(phi int, F1, F2, ReH, ReE, ReHtilde float64) float64 {
	phiIndex := t.getPhiIndex(phi)

	// getBHUU
	AUUBH := (8. * t.M2) / (t.t * t.kqp[phiIndex] * t.kpqp[phiIndex]) * ((4. * t.tau * (t.kP[phiIndex]*t.kP[phiIndex] + t.kpP[phiIndex]*t.kpP[phiIndex])) - ((t.tau + 1.) * (t.kd[phiIndex]*t.kd[phiIndex] + t.kpd[phiIndex]*t.kpd[phiIndex])))
	BUUBH := (16. * t.M2) / (t.t * t.kqp[phiIndex] * t.kpqp[phiIndex]) * (t.kd[phiIndex]*t.kd[phiIndex] + t.kpd[phiIndex]*t.kpd[phiIndex])

	con_AUUBH := AUUBH * GeV2nb * t.jcob
	con_BUUBH := BUUBH * GeV2nb * t.jcob

	bhAUU := (t.Gamma / t.t) * con_AUUBH * (F1*F1 + t.tau*F2*F2)
	bhBUU := (t.Gamma / t.t) * con_BUUBH * (t.tau * (F1 + F2) * (F1 + F2))

	xbhUU := bhAUU + bhBUU

	// getIUU
	AUUI := -4. * math.Cos(float64(phi)*RAD) * (t.Dplus[phiIndex]*((t.kqp_T[phiIndex]-2.*t.kk_T-2.*t.kqp[phiIndex])*t.kpP[phiIndex]+(2.*t.kpqp[phiIndex]-2.*t.kkp_T-t.kpqp_T[phiIndex])*t.kP[phiIndex]+t.kpqp[phiIndex]*t.kP_T[phiIndex]+t.kqp[phiIndex]*t.kpP_T[phiIndex]-2.*t.kkp*t.kP_T[phiIndex]) -
		t.Dminus[phiIndex]*((2.*t.kkp-t.kpqp_T[phiIndex]-t.kkp_T)*t.Pqp[phiIndex]+2.*t.kkp*t.qpP_T[phiIndex]-t.kpqp[phiIndex]*t.kP_T[phiIndex]-t.kqp[phiIndex]*t.kpP_T[phiIndex]))
	BUUI := -2. * t.xi * math.Cos(float64(phi)*RAD) * (t.Dplus[phiIndex]*((t.kqp_T[phiIndex]-2.*t.kk_T-2.*t.kqp[phiIndex])*t.kpd[phiIndex]+(2.*t.kpqp[phiIndex]-2.*t.kkp_T-t.kpqp_T[phiIndex])*t.kd[phiIndex]+t.kpqp[phiIndex]*t.kd_T[phiIndex]+t.kqp[phiIndex]*t.kpd_T[phiIndex]-2.*t.kkp*t.kd_T[phiIndex]) -
		t.Dminus[phiIndex]*((2.*t.kkp-t.kpqp_T[phiIndex]-t.kkp_T)*t.qpd[phiIndex]+2.*t.kkp*t.qpd_T[phiIndex]-t.kpqp[phiIndex]*t.kd_T[phiIndex]-t.kqp[phiIndex]*t.kpd_T[phiIndex]))
	CUUI := -2. * math.Cos(float64(phi)*RAD) * (t.Dplus[phiIndex]*(2.*t.kkp*t.kd_T[phiIndex]-t.kpqp[phiIndex]*t.kd_T[phiIndex]-t.kqp[phiIndex]*t.kpd_T[phiIndex]+4.*t.xi*t.kkp*t.kP_T[phiIndex]-2.*t.xi*t.kpqp[phiIndex]*t.kP_T[phiIndex]-2.*t.xi*t.kqp[phiIndex]*t.kpP_T[phiIndex]) -
		t.Dminus[phiIndex]*(t.kkp*t.qpd_T[phiIndex]-t.kpqp[phiIndex]*t.kd_T[phiIndex]-t.kqp[phiIndex]*t.kpd_T[phiIndex]+2.*t.xi*t.kkp*t.qpP_T[phiIndex]-2.*t.xi*t.kpqp[phiIndex]*t.kP_T[phiIndex]-2.*t.xi*t.kqp[phiIndex]*t.kpP_T[phiIndex]))

	con_AUUI := AUUI * GeV2nb * t.jcob
	con_BUUI := BUUI * GeV2nb * t.jcob
	con_CUUI := CUUI * GeV2nb * t.jcob

	iAUU := (t.Gamma / (math.Abs(t.t) * t.QQ)) * con_AUUI * (F1*ReH + t.tau*F2*ReE)
	iBUU := (t.Gamma / (math.Abs(t.t) * t.QQ)) * con_BUUI * (F1 + F2) * (ReH + ReE)
	iCUU := (t.Gamma / (math.Abs(t.t) * t.QQ)) * con_CUUI * (F1 + F2) * ReHtilde

	xIUU := iAUU + iBUU + iCUU

	// getBHUU + getIUU = xbhUU + (-1 * xIUU)
	return xbhUU - xIUU
}
