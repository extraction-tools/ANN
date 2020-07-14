/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Calculation of the elastic form factors using the Galster parametrization  //
//                                                                             //
//  Calculates GM, GE, F1, F2, GA, and GP and from their t dependence          //
//                                                                             //
//  Written as part of the work referencing arXiv: 1903.05742                  //
//                                                                             //
//  Written by: Brandon Kriesten                                               //
//                                                                             //
//  Email: btk8bh@virginia.edu                                                 //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include "TFormFactors.h"

using namespace std;  	// std namespace: so you can do things like 'cout'

ClassImp(TFormFactors)				// classimp: necessary for root


Double_t TFormFactors::ffGE(Double_t t) {
  Double_t GE = 1.0 / ( 1.0 + ( -t / 0.710649 ) ) / ( 1.0 + ( -t / 0.710649 ) );
  return GE;
}

Double_t TFormFactors::ffGM(Double_t t) {
  Double_t shape = ffGE(t);
  Double_t GM0 = 2.792847337;
  return GM0*shape;
}

Double_t TFormFactors::ffF2(Double_t t) {
  Double_t f2 = (ffGM(t) - ffGE(t)) / (1. - t / (4.*.938272*.938272));
  return f2;
}

Double_t TFormFactors::ffF1(Double_t t) {
  Double_t f1 = ffGM(t)- ffF2(t);
  return f1;
}

Double_t TFormFactors::ffGA(Double_t t) {
  Double_t ga = 1.2695;
  Double_t ma = 1.026;
  Double_t part = t/(ma*ma);
  Double_t dif = (1-part)*(1-part);
  Double_t GA = ga/dif;
  return GA;
}

Double_t TFormFactors::ffGP(Double_t t) {
  return 8.13;
}
