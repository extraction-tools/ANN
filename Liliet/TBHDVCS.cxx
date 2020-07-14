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
  jcob = 1./ ( 2. * M * x * K(3) ) * 2. * PI * 2.;

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

  // Transverse vector products defined after eq.(241c) -----------------------
	//kk_T   = 0.5 * ( e / ( 1 - e ) ) * QQ;
	//kqp_T  = ( QQ / ( sqrt( gg ) * sqrt( 1 + gg ) ) ) * sqrt ( (0.5 * e) / ( 1 - e ) ) * ( 1. + x * t / QQ ) * sin(theta) * cos( phi * RAD );
	//dd_T   = ( 1. - xi * xi ) * ( tmin - t );
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
Double_t TBHDVCS::GetDVCSUUxs(Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  FUUT = 4.* ( ( 1 - xi * xi ) * ( ReH * ReH + ImH * ImH + ReHtilde * ReHtilde + ImHtilde * ImHtilde ) + ( tmin - t ) / ( 2.* M2 ) * ( ReE * ReE + ImE * ImE + xi * xi * ReEtilde * ReEtilde + xi * xi * ImEtilde * ImEtilde )
         - ( 2.* xi * xi ) / ( 1 - xi * xi ) * ( ReH * ReE + ImH * ImE + ReHtilde * ReEtilde + ImHtilde * ImEtilde ) );

  xdvcsUU =  GeV2nb * jcob * Gamma / QQ / ( 1 - e ) * FUUT;

  return xdvcsUU;
}

//====================================================================================
// Polarized Beam Polarized Target DVCS Cross Section
//====================================================================================
Double_t TBHDVCS::GetDVCSLLxs(Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  FLL = 4.* ( 2.* ( 1 - xi * xi ) * ( ReH * ReHtilde + ImH * ImHtilde ) + 2.* ( tmin - t ) / ( 2.* M2 ) * ( ReE * xi * ReEtilde + ImE * xi * ImEtilde )
        + ( 2.* xi * xi ) / ( 1 - xi * xi ) * ( ReH * ReEtilde + ImH * ImEtilde + ReHtilde * ReE + ImHtilde * ImE ) );

  xdvcsLL =  GeV2nb * jcob * Gamma / QQ / ( 1 - e ) * 2.* sqrt(1 - e * e) * FLL;

  return xdvcsLL;
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
  // I multiply by 2 because I think Auu and Buu are missing a factor 2
  con_AUUBH = 2. * AUUBH * GeV2nb * jcob;
  con_BUUBH = 2. * BUUBH * GeV2nb * jcob;

  // Unpolarized Coefficients multiplied by the Form Factors
  bhAUU = (Gamma/t) * con_AUUBH * ( F1 * F1 + tau * F2 * F2 );
  bhBUU = (Gamma/t) * con_BUUBH * ( tau * ( F1 + F2 ) * ( F1 + F2 ) );

  // Unpolarized BH cross section
  xbhUU = bhAUU + bhBUU;

  return xbhUU;
}
//====================================================================================
// Polarized Beam Polarized Target BH Cross Section
//====================================================================================
Double_t TBHDVCS::GetBHLLxs(Double_t F1, Double_t F2) {

  // Coefficients of the BH longitudinal polarized beam longitudinal polarized target structure function FLL_BH
  ALLBH = - (16. * M2) / (t * kqp * kpqp) * ( (ppSL / M) * ( kpd * kpd - kd * kd - 2. * tau * ( kpd * kpp - kd * kp ) ) + t * (kSL / M) * ( 1. + tau ) * kd - t * (kpSL/M) * ( 1. + tau ) * kpd );  // eq. 155
  BLLBH = (16. * M2) / (t * kqp * kpqp) * ( (ppSL / M) * ( kpd * kpd - kd * kd ) +  t * (kSL / M) * kd - t * (kpSL/M) * kpd ); // eq. 156

  // Convert Coefficients to nano-barn and use Defurne's Jacobian
  con_ALLBH = ALLBH * GeV2nb * jcob;
  con_BLLBH = BLLBH * GeV2nb * jcob;

  // Coefficients multiplied by the Form Factors (eq. 154)
  bhALL = (Gamma/t) * con_ALLBH * F2 * ( F1 + F2 );
  bhBLL = (Gamma/t) * con_BLLBH * ( F1 + F2 ) * ( F1 + F2 );

  // BH longitudinal polarized beam longitudinal polarized target cross Section
  xbhLL = bhALL + bhBLL;

  return xbhLL;
}
//====================================================================================
// Unpolarized BH-DVCS Interference Cross Section
//====================================================================================
Double_t TBHDVCS::GetIUUxs(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde,
                           Double_t &con_AUUI, Double_t &con_BUUI, Double_t &con_CUUI) {

  // Get the 4-vector products. Note: Kinematics should have been previously set.
  Set4VectorsPhiDep(phi);
  Set4VectorProducts(phi);

  // Interference coefficients given on eq. (241a,b,c)--------------------
  AUUI = -4.0 / (kqp * kpqp) * ( ( QQ + t ) * ( 2.0 * ( kP + kpP ) * kk_T   + ( Pq * kqp_T ) + 2.* ( kpP * kqp ) - 2.* ( kP * kpqp ) + kpqp * kP_T + kqp * kpP_T - 2.*kkp * kP_T ) +
                                                    ( QQ - t + 4.* kd ) * ( Pqp * ( kkp_T + kqp_T - 2.* kkp )  + 2.* kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) );
  // BUUI = 2.0 * xi / ( kqp * kpqp) * ( ( QQ + t ) * ( 2.* kk_T * ( kd + kpd ) + kqp_T * ( qd - kqp - kpqp + 2.*kkp ) + 2.* kqp * kpd - 2.* kpqp * kd ) +
  //                                                        ( QQ - t + 4.* kd ) * ( ( kk_T - 2.* kkp ) * qpd - kkp * dd_T - 2.* kd_T * kqp ) ) / tau;
  BUUI = 2.0 * xi / ( kqp * kpqp) * ( ( QQ + t ) * ( 2.* kk_T * ( kd + kpd ) + kqp_T * ( qd - kqp - kpqp + 2.*kkp ) + 2.* kqp * kpd - 2.* kpqp * kd ) +
                                                    ( QQ - t + 4.* kd ) * ( ( kk_T - 2.* kkp ) * qpd - kkp * dd_T - 2.* kd_T * kqp ) );
  CUUI = 2.0 / ( kqp * kpqp) * ( -1. * ( QQ + t ) * ( 2.* kkp - kpqp - kqp + 2.* xi * (2.* kkp * kP_T - kpqp * kP_T - kqp * kpP_T) ) * kd_T +
                                                  ( QQ - t + 4.* kd ) * ( ( kqp + kpqp ) * kd_T + dd_T * kkp + 2.* xi * ( kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) );

  // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
  con_AUUI = AUUI * GeV2nb * jcob;
  con_BUUI = BUUI * GeV2nb * jcob;
  con_CUUI = CUUI * GeV2nb * jcob;

  //Unpolarized Coefficients multiplied by the Form Factors
  iAUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_AUUI * ( F1 * ReH + tau * F2 * ReE );
  //iBUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_BUUI * tau * ( F1 + F2 ) * ( ReH + ReE );
  iBUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_BUUI * ( F1 + F2 ) * ( ReH + ReE );
  iCUU = (Gamma/(-t * QQ)) * cos( phi * RAD ) * con_CUUI * ( F1 + F2 ) * ReHtilde;

  // Unpolarized BH-DVCS interference cross section
  xIUU = iAUU + iBUU + iCUU;
  return xIUU;
}
//====================================================================================
// Unpolarized Beam Polarized Target BH-DVCS Interference Cross Section
//====================================================================================
Double_t TBHDVCS::GetIULxs(Double_t phi, Double_t F1, Double_t F2, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  //Calculates the Unpolarized Beam Polarized Target Coefficients
  //From Brandon's code
  AULI = 16.* Dplus * ( kpP * ( 2.* kk_T - kqp_T + 2.* kqp ) + kP * ( 2.* kkp_T - kpqp_T + 2.* kpqp ) + 2.* kkp * kP_T - kpqp * kP_T - kqp * kpP_T ) * sin(phi * RAD)
       + 16.* Dminus * ( Pqp * ( kkp_T + kpqp_T - 2.*kkp ) - ( 2.* kkp * qpP_T - kpqp * kP_T - kqp * kpP_T) )* sin(phi * RAD);
  BULI = 8.* xi * Dplus * ( kpd * ( 2.* kk_T - kqp_T + 2.* kqp ) + kd * ( 2.* kkp_T - kpqp_T + 2.* kpqp ) + 2.* kkp * kpd_T - kpqp * kd_T - kqp * kpd_T ) * sin(phi * RAD)
       + 8.* xi * Dminus * ( qpd * ( kkp_T + kpqp_T - 2.* kkp ) - ( 2.* kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) ) * sin(phi * RAD);
  CULI = 4.* Dplus * ( 2.* ( 2.* kkp * kd_T - kpqp * kd_T - kqp * kpd_T ) + 4.* xi * ( 2.* kkp * kP_T - kpqp * kP_T - kqp * kpP_T ) ) * sin(phi * RAD)
       + 4.* Dminus * ( -2 * ( kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) - 4.* xi * ( kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) * sin(phi * RAD);
  //From Paper
  // AULI = -16.* Dplus * ( 2.* kk_T * ( kpP + kP ) - kd_T * Pq + 2. * kqp * kpP - 2. * kpqp * kP ) - 16. * Dminus * ( kk_T - kd_T + 2. * kqp ) * kpP;
  // BULI = -8.* xi * Dplus * ( 2.* kk_T * ( kpd + kd ) - kd_T * ( qd + 2.* kkp - kpqp - kqp ) + 2.* kqp * kpd - 2.* kpqp * kd ) - 8.* xi * Dminus * ( kk_T * qpd - 2.* kkp * qpd - kkp * dd_T - 2.* kd_T * kqp );
  // CULI = -8.* Dplus * ( 2.* kkp - kpqp - kqp ) * kd_T - 8.* Dminus * ( kkp * dd_T + kd_T * ( kpqp + kqp ) );

  // Convert Coefficients to nano-barn and use Defurne's Jacobian
  con_AULI = AULI * GeV2nb * jcob;
  con_BULI = BULI * GeV2nb * jcob;
  con_CULI = CULI * GeV2nb * jcob;

  // Coefficients multiplied by the Form Factors
  iAUL = (Gamma/(-t * QQ)) * con_AULI * ( F1 * ( ImHtilde - xi * ImEtilde ) + tau * F2 * ImEtilde );
  iBUL = (Gamma/(-t * QQ)) * con_BULI * ( F1 + F2 ) * ImHtilde;
  iCUL = (Gamma/(-t * QQ)) * con_CULI * ( F1 + F2 ) * ( ImH + ImE );

  // Unpolarized Beam Polarized Target BH-DVCS interference cross section
  xIUL = 2. * ( iAUL + iBUL + iCUL );

  return xIUL;
}
//====================================================================================
// Polarized Beam Unpolarized Target BH-DVCS Interference Cross Section
//====================================================================================
Double_t TBHDVCS::GetILUxs(Double_t phi, Double_t F1, Double_t F2, Double_t ImH, Double_t ImE, Double_t ImHtilde) {

  //Calculates the Polarized Beam Unpolarized Target Coefficients
  ALUI = 16.* Dplus * ( 2.* ( K.Px() * Pplus * KP.Px() * kminus - K.Px() * Pplus * kpminus * K.Px() + K.Px() * Pminus * kpplus * K.Px() - K.Px() * Pminus * KP.Px() * kplus + K.Px() * P.Px() * kpminus * kplus - K.Px() * P.Px() * kpplus * kminus ) +
         KP.Px() * Pplus * qpminus * K.Px() - KP.Px() * Pplus * QP.Px() * kminus + KP.Px() * Pminus * QP.Px() * kplus - KP.Px() * Pminus * qpplus * K.Px() + KP.Px() * P.Px() * qpplus * kminus - KP.Px() * P.Px() * qpminus * kplus + K.Px() * Pplus * qpminus * KP.Px() -
         K.Px() * Pplus * QP.Px() * kpminus + K.Px() * Pminus * QP.Px() * kpplus - K.Px() * Pminus * qpplus * KP.Px() + K.Px() * P.Px() * qpplus * kpminus - K.Px() * P.Px() * qpminus * kpplus + 2.* ( qpminus * Pplus - qpplus * Pminus ) * kkp ) * sin(phi * RAD) +
         16.* Dminus * ( 2.* ( kminus * kpplus - kplus * kpminus ) * Pqp + kpminus * kplus * QP.Px() * P.Px() + kpplus * K.Px() * qpminus * P.Px() + KP.Px() * kminus * qpplus * P.Px() - kpplus * kminus * QP.Px() * P.Px() - KP.Px() * kplus * qpminus * P.Px() -
         kpminus * K.Px() * qpplus * P.Px() + kpminus * kplus * QP.Py() * P.Py() - kpplus * kminus * QP.Py() * P.Py() ) * sin(phi * RAD);
  BLUI = 8.* xi * Dplus * ( 2.* ( K.Px() * dplus * KP.Px() * kminus - K.Px() * dplus * kpminus * K.Px() + K.Px() * dminus * kpplus * K.Px() - K.Px() * dminus * KP.Px() * kplus + K.Px() * D.Px() * kpminus * kplus - K.Px() * D.Px() * kpplus * kminus ) +
         KP.Px() * dplus * qpminus * K.Px() - KP.Px() * dplus * QP.Px() * kminus + KP.Px() * dminus * QP.Px() * kplus - KP.Px() * dminus * qpplus * K.Px() + KP.Px() * D.Px() * qpplus * kminus - KP.Px() * D.Px() * qpminus * kplus + K.Px() * dplus * qpminus * KP.Px() -
         K.Px() * dplus * QP.Px() * kpminus + K.Px() * dminus * QP.Px() * kpplus - K.Px() * dminus * qpplus * KP.Px() + K.Px() * D.Px() * qpplus * kpminus - K.Px() * D.Px() * qpminus * kpplus + 2.* ( qpminus * dplus - qpplus * dminus ) * kkp ) * sin(phi * RAD) +
         8.* xi * Dminus * ( 2.* ( kminus * kpplus - kplus * kpminus ) * qpd + kpminus * kplus * QP.Px() * D.Px() + kpplus * K.Px() * qpminus * D.Px() + KP.Px() * kminus * qpplus * D.Px() - kpplus * kminus * QP.Px() * D.Px() - KP.Px() * kplus * qpminus * D.Px() -
         kpminus * K.Px() * qpplus * D.Px() + kpminus * kplus * QP.Py() * D.Py() - kpplus * kminus * QP.Py() * D.Py() ) *sin(phi * RAD);
  CLUI = -8.* Dplus * ( 2. * ( KP.Px() * kpminus * kplus * D.Px() - KP.Px() * kpplus * kminus * D.Px() ) + KP.Px() * qpminus * kplus * D.Px() - KP.Px() * qpplus * kminus * D.Px() + K.Px() * qpminus * kpplus * D.Px() - K.Px() * qpplus * kpminus * D.Px() ) * sin(phi * RAD) -
         8.* Dminus * ( -kpminus * K.Px() * qpplus * D.Px() + kpminus * kplus * QP.Px() * D.Px() + kpplus * K.Px() * qpminus * D.Px() - kpplus * kminus * QP.Px() * D.Px() + KP.Px() * kminus * qpplus * D.Px() - KP.Px() * kplus * qpminus * D.Px() - QP.Py() * D.Py() * ( kpplus * kminus - kpminus * kplus ) ) * sin(phi * RAD);

 // Convert Coefficients to nano-barn and use Defurne's Jacobian
 con_ALUI = ALUI * GeV2nb * jcob;
 con_BLUI = BLUI * GeV2nb * jcob;
 con_CLUI = CLUI * GeV2nb * jcob;

 // Coefficients multiplied by the Form Factors
 iALU = (Gamma/(-t * QQ)) * con_ALUI * ( F1 * ImH + tau * F2 * ImE );
 iBLU = (Gamma/(-t * QQ)) * con_BLUI * ( F1 + F2 ) * ( ImH + ImE );
 iCLU = (Gamma/(-t * QQ)) * con_CLUI * ( F1 + F2 ) * ImHtilde;

 // Polarized Beam Unpolarized Target BH-DVCS interference cross section
 xILU = 2. * ( iALU + iBLU + iCLU );

 return xILU;
}

//====================================================================================
// Polarized Beam Polarized Target BH-DVCS Interference Cross Section
//====================================================================================
Double_t TBHDVCS::GetILLxs(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde) {

  //Calculates the Longitudinally Polarized Coefficients in front of the EFFs
  ALLI = 16.* Dplus * ( 2.* KP.Px() * ( KP.Px() * kminus - kpminus * K.Px() ) * Pplus + 2.* KP.Px() * ( kpplus * K.Px() - KP.Px() * kplus ) * Pminus + 2.* KP.Px() * ( kpminus * kplus - kpplus * kminus ) * P.Px() + KP.Px() * ( qpminus * K.Px() - QP.Px() * kminus ) * Pplus +
         KP.Px() * ( QP.Px() * kplus - qpplus * K.Px() ) * Pminus + KP.Px() * ( qpplus * kminus - qpminus * kplus ) * P.Px() + K.Px() * ( qpminus * KP.Px() - QP.Px() * kpminus ) * Pplus + K.Px() * ( QP.Px() * kpplus - qpplus * KP.Px() ) * Pminus + K.Px() * ( qpplus * kpminus - qpminus * kpplus ) * P.Px() - 2.* kkp * ( qpplus * Pminus - qpminus * Pplus ) ) * cos(phi * RAD) +
         16.* Dminus * ( 2.* Pqp * ( kpplus * kminus - kpminus * kplus ) + P.Py() * QP.Py() * ( kpplus * kminus - kpminus * kplus ) + P.Px() * ( KP.Px() * kplus * qpminus - KP.Px() * kminus * qpplus + kpminus * K.Px() * qpplus - kpminus * kplus * QP.Px() + kpplus * kminus * QP.Px() - kpplus * K.Px() * qpminus ) ) * cos(phi * RAD);
  BLLI = 8.* xi * Dplus * ( 2.* KP.Px() * ( KP.Px() * kminus - kpminus * K.Px() ) * dplus + 2.* KP.Px() * ( kpplus * K.Px() - KP.Px() * kplus ) * dminus + 2.* KP.Px() * ( kpminus * kplus - kpplus * kminus ) * D.Px() + KP.Px() * ( qpminus * K.Px() - QP.Px() * kminus ) * dplus + KP.Px() * ( QP.Px() * kplus - qpplus * K.Px() ) * dminus +
         KP.Px() * D.Px() * ( qpplus * kminus - qpminus * kplus ) + K.Px() * ( qpminus * KP.Px() - QP.Px() * kpminus ) * dplus + K.Px() * ( QP.Px() * kpplus - qpplus * KP.Px() ) * dminus + K.Px() * ( qpplus * kpminus -qpminus * kpplus ) * D.Px() - 2.* kkp * ( qpplus * dminus - qpminus * dplus ) ) * cos(phi * RAD) +
         8.* xi * Dminus * ( 2.* qpd * ( kpplus * kminus - kpminus * kplus ) + D.Py() * QP.Py() * ( kpplus * kminus - kpminus * kplus ) + D.Px() * ( KP.Px() * kplus * qpminus - KP.Px() * kminus * qpplus + kpminus * K.Px() * qpplus - kpminus * kplus * QP.Px() + kpplus * kminus * QP.Px() - kpplus * K.Px() * qpminus ) ) * cos(phi * RAD);
  CLLI = 16.* Dplus * ( 2.* ( K.Px() * kminus * kpplus * D.Px() - K.Px() * kpminus * kplus * D.Px() ) + KP.Px() * qpplus * kminus * D.Px() - KP.Px() * qpminus * kplus * D.Px() + K.Px() * qpplus * kpminus * D.Px() - K.Px() * qpminus * kpplus * D.Px() ) * cos(phi * RAD) -
         16.* Dminus * ( -D.Px() * ( kpminus * kplus * QP.Px() - kpminus * K.Px() * qpplus + kpplus * K.Px() * qpminus - kpplus * kminus * QP.Px() + KP.Px() * kminus * qpplus - KP.Px() * kplus * qpminus ) + QP.Px() * kpplus * kminus * D.Py() - QP.Py() * kpminus * kplus * D.Py() ) * cos(phi * RAD);

  // Convert Coefficients to nano-barn and use Defurne's Jacobian
  con_ALLI = ALLI * GeV2nb * jcob;
  con_BLLI = BLLI * GeV2nb * jcob;
  con_CLLI = CLLI * GeV2nb * jcob;

  // Coefficients multiplied by the Form Factors
  iALL = (Gamma/(-t * QQ)) * con_ALLI * ( F1 * ( ReHtilde - xi * ReEtilde ) + tau * F2 * ReEtilde );
  iBLL = (Gamma/(-t * QQ)) * con_BLLI * ( F1 + F2 ) * ReHtilde;
  iCLL = (Gamma/(-t * QQ)) * con_CLLI * ( F1 + F2 ) * ( ReH + ReE );

  // Polarized Beam Unpolarized Target BH-DVCS interference cross section
  xILL = 4. * ( iALL + iBLL + iCLL );

  return xILL;
}
//====================================================================================
// Unpolarized Beam Polarized Target Beam Asymmetry
//====================================================================================
Double_t TBHDVCS::GetAsymUL(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  // Asymmetry UL (sigma_UL / sigma_UU)
  asym_ul = GetIULxs(phi, F1, F2, ImH, ImE, ImHtilde, ImEtilde) / ( GetDVCSUUxs(F1, F2, ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde) + GetBHUUxs(phi, F1, F2) + GetIUUxs(phi, F1, F2, ReH, ReE, ReHtilde, con_AUUI, con_BUUI, con_CUUI) );

  return asym_ul;
}
//====================================================================================
// Polarized Beam Unpolarized Target Beam Asymmetry
//====================================================================================
Double_t TBHDVCS::GetAsymLU(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  // Asymmetry UL (sigma_UL / sigma_UU)
  asym_lu = GetILUxs(phi, F1, F2, ImH, ImE, ImHtilde) / ( GetDVCSUUxs(F1, F2, ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde) + GetBHUUxs(phi, F1, F2) + GetIUUxs(phi, F1, F2, ReH, ReE, ReHtilde, con_AUUI, con_BUUI, con_CUUI) );

  return asym_lu;
}
//====================================================================================
// Polarized Beam Polarized Target Beam Asymmetry
//====================================================================================
Double_t TBHDVCS::GetAsymLL(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, Double_t ImHtilde, Double_t ImEtilde) {

  // Asymmetry UL (sigma_UL / sigma_UU)
  asym_ll = ( GetDVCSLLxs(F1, F2, ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde) + GetBHLLxs(F1, F2) + GetILLxs(phi, F1, F2, ReH, ReE, ReHtilde, ReEtilde) ) /
            ( GetDVCSUUxs(F1, F2, ReH, ReE, ReHtilde, ReEtilde, ImH, ImE, ImHtilde, ImEtilde) + GetBHUUxs(phi, F1, F2) + GetIUUxs(phi, F1, F2, ReH, ReE, ReHtilde, con_AUUI, con_BUUI, con_CUUI) );

  return asym_ll;
}
