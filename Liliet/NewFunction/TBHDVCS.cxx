#include "TBHDVCS.h"

using namespace std;  	// std namespace: so you can do things like 'cout'

ClassImp(TBHDVCS)				// classimp: necessary for root


//____________________________________________________________________________________
TBHDVCS::TBHDVCS() {
	// Default Constructor
}
//____________________________________________________________________________________
TBHDVCS::~TBHDVCS() {
	// Default Destructor
}
//____________________________________________________________________________________
Double_t TBHDVCS::TProduct(TLorentzVector v1, TLorentzVector v2) {
	// Transverse product
  Double_t tv1v2;
  return tv1v2 = v1.Px() * v2.Px() + v1.Py() * v2.Py();
}
//____________________________________________________________________________________
void TBHDVCS::SetKinematics(Double_t _QQ, Double_t _x, Double_t _t, Double_t _k){

  QQ = _QQ; //Q^2 value
  x = _x;   //Bjorken x
  t = _t;   //momentum transfer squared
  k = _k;   //Electron Beam Energy
  M2 = M*M; //Mass of the proton  squared in GeV
  //fractional energy of virtual photon
  y = QQ / ( 2. * M * k * x ); // From eq. (23) where gamma is substituted from eq (12c)
  //squared gamma variable ratio of virtuality to energy of virtual photon
	gg = 4. * M2 * x * x / QQ; // This is gamma^2 [from eq. (12c)]
  //ratio of longitudinal to transverse virtual photon flux
	e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); // epsilon eq. (32)
  //Skewness parameter
	xi = 1. * x * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - x + x * t / QQ ) ); // skewness parameter eq. (12b) note: there is a minus sign on the write up that shouldn't be there
  //Minimum t value
  tmin = ( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( x * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* x ) ) ); // minimum t eq. (29)
  //Final Lepton energy
  kpr = k * ( 1. - y ); // k' from eq. (23)
  //outgoing photon energy
  qp = t / 2. / M + k - kpr; //q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
  //Final proton Energy
  po = M - t / 2. / M; // This is p'_0 from eq. (28b)
  pmag = sqrt( ( -t ) * ( 1. - t / 4. / M / M ) ); // p' magnitude from eq. (28b)
  //Angular Kinematics of outgoing photon
  cth = -1. / sqrt( 1. + gg ) * ( 1. + gg / 2. * ( 1. + t / QQ ) / ( 1. + x * t / QQ ) ); // This is cos(theta) eq. (26)
	theta = acos(cth); // theta angle
  //Lepton Angle Kinematics of initial lepton
  sthl = sqrt( gg ) / sqrt( 1. + gg ) * ( sqrt ( 1. - y - y * y * gg / 4. ) ); // sin(theta_l) from eq. (22a)
	cthl = -1. / sqrt( 1. + gg ) * ( 1. + y * gg / 2. ) ; // cos(theta_l) from eq. (22a)
  //ratio of momentum transfer to proton mass
  tau = -0.25 * t / M2;

  // phi independent 4 - momenta vectors defined on eq. (21) -------------
  K.SetPxPyPzE( k * sthl, 0.0, k * cthl, k );
  KP.SetPxPyPzE( K(0), 0.0, k * ( cthl + y * sqrt( 1. + gg ) ), kpr );
  Q = K - KP;
  p.SetPxPyPzE(0.0, 0.0, 0.0, M);

  // Sets the Mandelstam variable s which is the center of mass energy
  s = (p + K) * (p + K);

  // The Gamma factor in front of the cross section
  Gamma = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / ( s - M2 ) / ( s - M2 ) / sqrt( 1. + gg ) / x;

  // Defurne's Jacobian
  jcob = 2. * PI ;
}
//___________________________________________________________________________________
Double_t TBHDVCS::GetGamma() {

  return Gamma;
}
//___________________________________________________________________________________
void TBHDVCS::Set4VectorsPhiDep(Double_t phi) {

  // phi dependent 4 - momenta vectors defined on eq. (21) -------------
	QP.SetPxPyPzE(qp * sin(theta) * cos( phi * RAD ), qp * sin(theta) * sin( phi * RAD ), qp * cos(theta), qp);
  D = Q - QP; // delta vector eq. (12a)
  TLorentzVector pp = p + D; // p' from eq. (21)
  P = p + pp;
  P.SetPxPyPzE(.5*P.Px(), .5*P.Py(), .5*P.Pz(), .5*P.E());
}
//____________________________________________________________________________________
void TBHDVCS::Set4VectorProducts(Double_t phi) {

  // 4-vectors products (phi - independent)
  kkp  = K * KP;   //(kk')
  kq   = K * Q;    //(kq)
  kp   = K * p;    //(pk)
  kpp  = KP * p;   //(pk')

  // 4-vectors products (phi - dependent)
  kd   = K * D;    //(kΔ)
  kpd  = KP * D;   //(k'Δ)
  kP   = K * P;    //(kP)
  kpP  = KP * P;   //(k'P)
  kqp  = K * QP;   //(kq')
  kpqp = KP * QP;  //(k'q')
  dd   = D * D;    //(ΔΔ)
  Pq   = P * Q;    //(Pq)
  Pqp  = P * QP;   //(Pq')
  qd   = Q * D;    //(qΔ)
  qpd  = QP * D;   //(q'Δ)

  // Transverse vector products -----------------------
	kk_T   = TProduct(K,K);
  kkp_T  = kk_T;
  kqp_T  = TProduct(K,QP);
  kd_T   = -1.* kqp_T;
  dd_T   = TProduct(D,D);
  kpqp_T = TProduct(KP,QP);
  kP_T   = TProduct(K,P);
  kpP_T  = TProduct(KP,P);
  qpP_T  = TProduct(QP,P);
  kpd_T  = TProduct(KP,D);
  qpd_T  = TProduct(QP,D);

  // Invariants involving the spin 4-vector SL used for polarized BH cross section calculation
  ppSL  = ( 2. * M ) / ( sqrt( 1 + gg ) ) * ( x * ( 1 - ( t / QQ ) ) -  t / ( 2 * M2 ) ); // eq. 160
  kSL   = ( 2. * QQ ) / ( sqrt( 1 + gg ) ) * ( 1. + .5 * y * gg ) * ( 1. / ( 2. * M * x * y ) );  // eq. 161
  kpSL  = ( 2. * QQ ) / ( sqrt( 1 + gg ) ) * ( 1. - y - .5 * y * gg ) * ( 1. / ( 2. * M * x * y ) );  // eq. 162

  //Expressions that appear in the polarized interference coefficient calculations
  Dplus   = .5 / kpqp - .5 / kqp;
  Dminus  = .5 / kpqp + .5 / kqp;

  //Light cone variables expressed as A^{+-} = 1/sqrt(2)(A^{0} +- A^{3}) used on the polarized interference coefficients
  kplus   = ( K.E() + K.Pz() ) / sqrt(2);
  kminus  = ( K.E() - K.Pz() ) / sqrt(2);
  kpplus  = ( KP.E() + KP.Pz() ) / sqrt(2);
  kpminus = ( KP.E() - KP.Pz() ) / sqrt(2);
  qplus   = ( Q.E() + Q.Pz() ) / sqrt(2);
  qminus  = ( Q.E() - Q.Pz() ) / sqrt(2);
  qpplus  = ( QP.E() + QP.Pz() ) / sqrt(2);
  qpminus = ( QP.E() - QP.Pz() ) / sqrt(2);
  Pplus   = ( P.E() + P.Pz() ) / sqrt(2);
  Pminus  = ( P.E() - P.Pz() ) / sqrt(2);
  dplus   = ( D.E() + D.Pz() ) / sqrt(2);
  dminus  = ( D.E() - D.Pz() ) / sqrt(2);
}
//====================================================================================
// DVCS Unpolarized Cross Section
//====================================================================================
Double_t TBHDVCS::GetDVCSUUxs( Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  FUUT = 4.* ( ( 1 - xi * xi ) * ( ReH * ReH + ImH * ImH + ReHtilde * ReHtilde + ImHtilde * ImHtilde ) + ( tmin - t ) / ( 2.* M2 ) * ( ReE * ReE + ImE * ImE + xi * xi * ReEtilde * ReEtilde + xi * xi * ImEtilde * ImEtilde )
         - ( 2.* xi * xi ) / ( 1 - xi * xi ) * ( ReH * ReE + ImH * ImE + ReHtilde * ReEtilde + ImHtilde * ImEtilde ) );

  xdvcsUU =  GeV2nb * jcob * Gamma / QQ / ( 1 - e ) * FUUT;

  return xdvcsUU;
}

//====================================================================================
// BH Unpolarized Cross Section
//====================================================================================
Double_t TBHDVCS::GetBHUUxs(Double_t phi, Double_t F1, Double_t F2) {

  // Get the 4-vector products. Note: Kinematics should have been previously set.
  Set4VectorsPhiDep(phi);
  Set4VectorProducts(phi);

  // Coefficients of the BH unpolarized structure function FUU_BH
  AUUBH = (8. * M2) / (t * kqp * kpqp) * ( (4. * tau * (kP * kP + kpP * kpP) ) - ( (tau + 1.) * (kd * kd + kpd * kpd) ) );  //eq. 147
  BUUBH = (16. * M2) / (t* kqp * kpqp) * (kd * kd + kpd * kpd); // eq. 148

  // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
  con_AUUBH = AUUBH * GeV2nb * jcob;
  con_BUUBH = BUUBH * GeV2nb * jcob;

  // Unpolarized Coefficients multiplied by the Form Factors
  bhAUU = (Gamma/t) * con_AUUBH * ( F1 * F1 + tau * F2 * F2 );
  bhBUU = (Gamma/t) * con_BUUBH * ( tau * ( F1 + F2 ) * ( F1 + F2 ) );

  // Unpolarized BH cross section
  xbhUU = bhAUU + bhBUU;

  return xbhUU;
}

//====================================================================================
// Unpolarized BH-DVCS Interference Cross Section
//====================================================================================
Double_t TBHDVCS::GetIUUxs(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde,
                           Double_t &con_AUUI, Double_t &con_BUUI, Double_t &con_CUUI) {

  // Get the 4-vector products. Note: Kinematics should have been previously set.
  Set4VectorsPhiDep(phi);
  Set4VectorProducts(phi);

  // Interference coefficients given on eq. (238a,b,c)-------------------- last version
  AUUI = 1. / ( kqp * kpqp ) * ( ( QQ + t ) * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpP + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kP + kpqp * kP_T + kqp * kpP_T - 2.*kkp * kP_T ) -
                               ( QQ - t + 4.* kd ) * ( ( 2. * kkp - kpqp_T - kkp_T ) * Pqp + 2. * kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) );

  BUUI = xi / ( 2. * kqp * kpqp ) * ( ( QQ + t ) * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpd + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kd + kpqp * kd_T + kqp * kpd_T - 2.*kkp * kd_T ) -
                                      ( QQ - t + 4.* kd ) * ( ( 2. * kkp - kpqp_T - kkp_T ) * qpd + 2. * kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) );


  CUUI = 1. / ( 2. * kqp * kpqp) * ( ( QQ + t ) * ( 2. * kkp * kd_T - kpqp * kd_T - kqp * kpd_T + 4. * xi * kkp * kP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) -
                                     ( QQ - t + 4.* kd ) * ( kkp * qpd_T - kpqp * kd_T - kqp * kpd_T + 2. * xi * kkp * qpP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) );

  // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
  con_AUUI = AUUI * GeV2nb * jcob;
  con_BUUI = BUUI * GeV2nb * jcob;
  con_CUUI = CUUI * GeV2nb * jcob;

  //Unpolarized Coefficients multiplied by the Form Factors
  iAUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_AUUI * ( F1 * ReH + tau * F2 * ReE );
  iBUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_BUUI * ( F1 + F2 ) * ( ReH + ReE );
  iCUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_CUUI * ( F1 + F2 ) * ReHtilde;

  // Unpolarized BH-DVCS interference cross section
  xIUU = iAUU + iBUU + iCUU;
  return xIUU;
}
