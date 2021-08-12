#ifndef TVA1_UU_H
#define TVA1_UU_H


class TVA1_UU {


private:

  Double_t ALP_INV = 137.0359998; // 1 / Electromagnetic Fine Structure Constant
  Double_t PI = 3.1415926535;
  //Double_t PI = TMath::Pi();
  Double_t RAD = PI / 180.;
	Double_t M = 0.938272; //Mass of the proton in GeV
  Double_t GeV2nb = .389379*1000000; // Conversion from GeV to NanoBarn

	Double_t QQ, x, t, k;
	Double_t y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau;
  Double_t s;     // Mandelstam variable s which is the center of mass energy
  Double_t Gamma; // Factor in front of the cross section
  Double_t jcob;  //Defurne's Jacobian

  // 4-momentum vectors
  TLorentzVector K, KP, Q, QP, D, p, P;
  // 4 - vector products independent of phi
  Double_t kkp, kq, kp, kpp;
  // 4 - vector products dependent of phi
  Double_t kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd;
  // Transverse 4-vector products
  Double_t kk_T, kqp_T, kkp_T, kd_T, dd_T, kpqp_T, kP_T, kpP_T, qpP_T, kpd_T, qpd_T;
  //Expressions that appear in the polarized interference coefficient calculations
  Double_t Dplus, Dminus;

  Double_t FUUT;
  Double_t AUUBH, BUUBH; // Coefficients of the BH unpolarized structure function FUU_BH
  Double_t AUUI, BUUI, CUUI; // Coefficients of the BHDVCS interference unpolarized structure function FUU_I
  Double_t con_AUUBH, con_BUUBH; // Coefficients times the conversion to nb and the jacobian
  Double_t con_AUUI, con_BUUI, con_CUUI;  // Coefficients times the conversion to nb and the jacobian
  Double_t bhAUU, bhBUU; // Auu and Buu term of the BH cross section
  Double_t iAUU, iBUU, iCUU; // Terms of the interference containing AUUI, BUUI and CUUI
  Double_t xdvcsUU, xbhUU, xIUU; // Unpolarized cross sections


public:

   TVA1_UU();  // Constructor
  ~TVA1_UU();  // Destructor

  Double_t TProduct(TLorentzVector v1, TLorentzVector v2);

  void SetKinematics(Double_t _QQ, Double_t _x, Double_t _t, Double_t _k);

  Double_t GetGamma();

  void Set4VectorsPhiDep(Double_t phi);

  void Set4VectorProducts(Double_t phi);

  Double_t GetDVCSUU(Double_t ReH, Double_t ReE, Double_t ReHtilde, Double_t ReEtilde, Double_t ImH, Double_t ImE, 		      Double_t ImHtilde, Double_t ImEtilde);

  Double_t GetBHUU(Double_t phi, Double_t F1, Double_t F2);

  Double_t GetIUU(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde);
  
  void GetIUUCoefficients(Double_t phi, Double_t &AUUI, Double_t &BUUI, Double_t &CUUI);
  
  Double_t GetIUULinear(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE);


	ClassDef(TVA1_UU,1);


};

#endif
