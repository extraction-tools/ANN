package main

import "os"

type TLorentzVector struct {
	x, y, z, t float64
}

func (l *TLorentzVector) parenthesesOperator(i int) float64 {
	switch i {
	case 0:
		return l.x
	case 1:
		return l.y
	case 2:
		return l.z
	case 3:
		return l.t
	default:
		os.Exit(1)
		return 0 // will never get here but Go requires it
	}
}

func (l *TLorentzVector) Px() float64 {
	return l.x
}

func (l *TLorentzVector) Py() float64 {
	return l.y
}

func (l *TLorentzVector) Pz() float64 {
	return l.z
}

func (l *TLorentzVector) E() float64 {
	return l.t
}

func (l TLorentzVector) plus(other TLorentzVector) TLorentzVector {
	return TLorentzVector{
		x: l.x + other.x,
		y: l.y + other.y,
		z: l.z + other.z,
		t: l.t + other.t,
	}
}

func (l TLorentzVector) minus(other TLorentzVector) TLorentzVector {
	return TLorentzVector{
		x: l.x - other.x,
		y: l.y - other.y,
		z: l.z - other.z,
		t: l.t - other.t,
	}
}

func (l TLorentzVector) multiply(other TLorentzVector) float64 {
	x := l.x * other.x
	y := l.y * other.y
	z := l.z * other.z
	t := l.t * other.t

	return t - z - y - x
}
