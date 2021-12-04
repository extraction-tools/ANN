#ifndef _TFORMFACTORS_H
#define _TFORMFACTORS_H

#include <math.h>

typedef struct {
	double M = 0.938272; //Mass of the proton in GeV

	// Kelly's parametrization fit Parameters
	double a1_GEp = -0.24;
	double b1_GEp = 10.98;
	double b2_GEp = 12.82;
	double b3_GEp = 21.97;
	double a1_GMp = 0.12;
	double b1_GMp = 10.97;
	double b2_GMp = 18.86;
	double b3_GMp = 6.55;


} TFormFactors;


// BBBA05 fit parameters
double ffGE(double t);
double ffGM(double t);
double ffF2(double t);
double ffF1(double t);
double ffGA(double t);
double ffGP(double t);

// Kelly parametrization
double ffGEp(TFormFactors *self, double t);
double ffGMp(TFormFactors *self, double t);
double ffF2_K(TFormFactors *self, double t);
double ffF1_K(TFormFactors *self, double t);


// ==================== Dipole parametrization used by Simonetta ===================
double ffGE(double t) {
  double GE = 1.0 / ( 1.0 + ( -t / 0.710649 ) ) / ( 1.0 + ( -t / 0.710649 ) );
  return GE;
}

double ffGM(double t) {
  double shape = ffGE(t);
  double GM0 = 2.792847337;
  return GM0*shape;
}

double ffF2(double t) {
  double f2 = (ffGM(t) - ffGE(t)) / (1. - t / (4.*.938272*.938272));
  return f2;
}

double ffF1(double t) {
  double f1 = ffGM(t)- ffF2(t);
  return f1;
}

double ffGA(double t) {
  double ga = 1.2695;
  double ma = 1.026;
  double part = t/(ma*ma);
  double dif = (1-part)*(1-part);
  double GA = ga/dif;
  return GA;
}

double ffGP(double t) {
  return 8.13;
}



// =============== Kelly parametrization ================
double ffGEp(TFormFactors *self, double t) {
  double tau = - t / 4. / self->M / self->M;
  double GEp = ( 1. + self->a1_GEp * tau )/( 1. + self->b1_GEp * tau + self->b2_GEp * tau * tau + self->b3_GEp * tau * tau * tau );
  return GEp;
}

double ffGMp(TFormFactors *self, double t) {
  double tau = - t / 4. / self->M / self->M;
  double GM0 = 2.7928473;
  double GMp = GM0 * ( 1. + self->a1_GMp * tau )/( 1. + self->b1_GMp * tau + self->b2_GMp * tau * tau + self->b3_GMp * tau * tau * tau );
  return GMp;
}

double ffF2_K(TFormFactors *self, double t) {
  double tau = - t / 4. / self->M / self->M;
  double f2_K = ( ffGMp(self, t) - ffGEp(self, t) ) / ( 1. + tau );
  return f2_K;
}

double ffF1_K(TFormFactors *self, double t) {
  double tau = - t / 4. / self->M / self->M;
  double f1_K = ( ffGMp(self, t) - ffF2_K(self, t) );
  return f1_K;
}


#endif