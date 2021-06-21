/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
//  Calculation of DVCS cross section using the BMK formulation of 2002 paper: //
//                                                                             //
//  arXiv:hep-ph/0112108v2                                                     //
//                                                                             //
//  Calculation is done up to Twist 2 approximation                            //
//                                                                             //
//  Written by: Liliet Calero Diaz                                             //
//                                                                             //
//  Email: lc2fc@virginia.edu                                                  //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////


#include "TBKM02.h"

using namespace std;  	// std namespace: so you can do things like 'cout'

ClassImp(TBKM02)				// classimp: necessary for root


//____________________________________________________________________________________
TBKM02::TBKM02() {
	// Default Constructor
}
//____________________________________________________________________________________
TBKM02::~TBKM02() {
	// Default Destructor
}
//___________________________________________________________________________________________________
TComplex TBKM02::cdstar( TComplex c, TComplex d ){ // ( C D* ) product

  TComplex dstar = TComplex::Conjugate(d);

  return ( c.Re() * dstar.Re() - c.Im() * dstar.Im() ) + ( c.Re() * dstar.Im() + c.Im() * dstar.Re() ) * TComplex::I();
}
//____________________________________________________________________________________
void TBKM02::SetKinematics(Double_t _QQ, Double_t _x, Double_t _t, Double_t _k){
  QQ = _QQ; //Q^2 value
  x = _x;   //Bjorken x
  t = _t;   //momentum transfer squared
  k = _k;   //Electron Beam energy
  ee = 4. * M2 * x * x / QQ; // epsilon squared
  y = sqrt(QQ) / ( sqrt(ee) * k );  // lepton energy fraction
  xi = x * ( 1. + t / 2. / QQ ) / ( 2. - x + x * t / QQ ); // Generalized Bjorken variable
  Gamma1 = x * y * y / ALP_INV / ALP_INV / ALP_INV / PI / 8. / QQ / QQ / sqrt( 1. + ee ); // factor in front of the cross section, eq. (22)
  s = 2. * M * k + M2;
  tmin = - QQ * ( 2. * ( 1. - x ) * ( 1. - sqrt(1. + ee) ) + ee ) / ( 4. * x * ( 1. - x ) + ee ); // eq. (31)
  K2 = - ( t / QQ ) * ( 1. - x ) * ( 1. - y - y * y * ee / 4.) * ( 1. - tmin / t ) * ( sqrt(1. + ee) + ( ( 4. * x * ( 1. - x ) + ee ) / ( 4. * ( 1. - x ) ) ) * ( ( t - tmin ) / QQ )  ); // eq. (30)
}
//___________________________________________________________________________________
void TBKM02::BHLeptonPropagators(Double_t phi) {

  // K*D 4-vector product (phi-dependent)
  KD = - QQ / ( 2. * y * ( 1. + ee ) ) * ( 1. + 2. * sqrt(K2) * cos( PI - ( phi * RAD ) ) - t / QQ * ( 1. - x * ( 2. - y ) + y * ee / 2. ) + y * ee / 2.  ); // eq. (29)

  // lepton BH propagators P1 and P2 (contaminating phi-dependence)
  P1 = 1. + 2. * KD / QQ;
  P2 = t / QQ - 2. * KD / QQ;
}
//___________________________________________________________________________________
Double_t TBKM02::BHUU(Double_t phi, Double_t F1, Double_t F2) { // BH Unpolarized Cross Section

  // Get BH propagators
  BHLeptonPropagators(phi);

  // BH unpolarized Fourier harmonics eqs. (35 - 37)
  c0_BH = 8. * K2 * ( ( 2. + 3. * ee ) * ( QQ / t ) * ( F1 * F1  - F2 * F2 * t / ( 4. * M2 ) ) + 2. * x * x * ( F1 + F2 ) * ( F1 + F2 ) ) +
          ( 2. - y ) * ( 2. - y ) * ( ( 2. + ee ) * ( ( 4. * x * x * M2 / t ) * ( 1. + t / QQ ) * ( 1. + t / QQ ) + 4. * ( 1. - x ) * ( 1. + x * t / QQ ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) +
          4. * x * x * ( x + ( 1. - x + ee / 2. ) * ( 1. - t / QQ ) * ( 1. - t / QQ ) - x * ( 1. - 2. * x ) * t * t / ( QQ * QQ ) ) * ( F1 + F2 ) * ( F1 + F2 ) ) +
          8. * ( 1. + ee ) * ( 1. - y - ee * y * y / 4. ) * ( 2. * ee * ( 1. - t / ( 4. * M2 ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) - x * x * ( 1. - t / QQ ) * ( 1. - t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) );

  c1_BH = 8. * sqrt(K2) * ( 2. - y ) * ( ( 4. * x * x * M2 / t - 2. * x - ee ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 2. * x * x * ( 1. - ( 1. - 2. * x ) * t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) );

  c2_BH = 8. * x * x * K2 * ( ( 4. * M2 / t ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 2. * ( F1 + F2 ) * ( F1 + F2 ) );

  // BH squared amplitude eq (25) divided by e^6
  Amp2_BH = 1. / ( x * x * y * y * ( 1. + ee ) * ( 1. + ee ) * t * P1 * P2 ) * ( c0_BH + c1_BH * cos( PI - (phi * RAD) ) + c2_BH * cos( 2. * ( PI - ( phi * RAD ) ) )  );

  Amp2_BH = GeV2nb * Amp2_BH; // convertion to nb

  return dsigma_BH = Gamma1 * Amp2_BH;
}
//___________________________________________________________________________________________________
Double_t TBKM02::DVCSUU( TComplex t2cffs[4] ) { // Pure DVCS Unpolarized Cross Section with just considering thre c0 term

  /* t2cffs = { H, E , Htilde, Etilde } Twist-2 Compton Form Factors*/
  TComplex H = t2cffs[0];
  TComplex E = t2cffs[1];
  TComplex Htilde = t2cffs[2];
  TComplex Etilde = t2cffs[3];

  // c coefficients (eq. 66) for pure DVCS .
  Double_t c_dvcs = 1./(2. - x)/(2. - x) * ( 4. * ( 1 - x ) * ( H.Rho2() + Htilde.Rho2() ) - x * x * ( cdstar(H, E) + cdstar(E, H) + cdstar(Htilde, Etilde) + cdstar(Etilde, Htilde) ) -
                    ( x * x + (2. - x) * (2. - x) * t / 4. / M2 ) * E.Rho2() - ( x * x * t / 4. / M2 ) * Etilde.Rho2() );

  // Pure DVCS unpolarized Fourier harmonics eqs. (43)
  c0_dvcs = 2. * ( 2. - 2. * y + y * y ) * c_dvcs;

  // DVCS squared amplitude eq (26) divided by e^6
  Amp2_DVCS = 1. / ( y * y * QQ ) *  c0_dvcs ;

  Amp2_DVCS = GeV2nb * Amp2_DVCS; // convertion to nb

  return dsigma_DVCS = Gamma1 * Amp2_DVCS;
}
//___________________________________________________________________________________________________
Double_t TBKM02::IUU(Double_t phi, Double_t F1, Double_t F2, TComplex t2cffs[4]) { // Interference Unpolarized Cross Section (writting it as in Liuti's comparison paper, i.e just c0 and c1 terms)

  // Get BH propagators
  BHLeptonPropagators(phi);

  /* t2cffs_I = { H, E , Htilde, Etilde } Twist-2 Compton Form Factors present in the interference term*/
  TComplex H = t2cffs[0];
  TComplex E = t2cffs[1];
  TComplex Htilde = t2cffs[2];
  TComplex Etilde = t2cffs[3]; // This CFF does not appear in the interference

  Double_t A, B, C; //Coefficients in from to the CFFs

  A = - 8. * K2 * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * ( 2. - y ) * ( 1. - y ) * ( 2. - x ) * t / QQ - 8. * sqrt(K2) * ( 2. - 2. * y + y * y ) * cos( PI - (phi * RAD) );
  B = 8. * x * x * ( 2. - y ) * (1 - y ) / ( 2. - x ) * t / QQ;
  C =  x / ( 2. - x ) * ( - 8. * K2 * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * sqrt(K2) * ( 2. - 2. * y + y * y ) * cos( PI - (phi * RAD) ) );

  // BH-DVCS interference squared amplitude eq (27) divided by e^6
  Amp2_I = 1. / ( x * y * y * y * t * P1 * P2 ) * ( A * ( F1 * H.Re() - t / 4. / M2 * F2 * E.Re() ) + B * ( F1 + F2 ) * ( H.Re() + E.Re() ) + C * ( F1 + F2 ) * Htilde.Re() );

  Amp2_I = GeV2nb * Amp2_I; // convertion to nb

  return dsigma_I = Gamma1 * Amp2_I;
}
