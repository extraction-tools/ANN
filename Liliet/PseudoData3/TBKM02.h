#ifndef TBKM02_H
#define TBKM02_H


class TBKM02 {


private:

	Double_t ALP_INV = 137.0359998; // 1 / Electromagnetic Fine Structure Constant
	Double_t PI = TMath::Pi();
	Double_t RAD = PI / 180.;
	Double_t M = 0.938272; //Mass of the proton in GeV
	Double_t M2 = M*M; //Mass of the proton  squared in GeV
	Double_t GeV2nb = .389379*1000000; // Conversion from 1/GeV2 to NanoBarn
	Double_t jcob;

	Double_t QQ, x, t, k;
	Double_t ee, y, xi, s, Gamma1, Gamma2, tmin, K2;
  	Double_t KD;
  	Double_t P1, P2;

  	Double_t c0_BH, c1_BH, c2_BH; // BH unpolarized c coefficients eqs. (35 - 37)
  	Double_t c0_dvcs, c1_dvcs; // DVCS unpolarized Fourier harmonics eqs. (43, 44)
	Double_t c0_I, c1_I, s1_I, c2_I, s2_I; // Interference unpolarized Fourier harmonics eqs. (53, 55)
  	Double_t Amp2_BH, Amp2_DVCS, Amp2_I; // squared amplitudes
  	Double_t dsigma_BH, dsigma_DVCS, dsigma_I; // differential cross sections

public:

	TBKM02();  // Constructor
  	~TBKM02(); // Destructor

  	TComplex cdstar( TComplex c, TComplex d ); // complex and complex conjugate numbers product
  	void SetKinematics(Double_t _QQ, Double_t _x, Double_t _t, Double_t _k);

  	void BHLeptonPropagators(Double_t phi);
  	Double_t BHUU(Double_t phi, Double_t F1, Double_t F2);  	
  	Double_t DVCSUU( TComplex t2cffs[4] );	
	Double_t IUU(Double_t phi, Double_t F1, Double_t F2, TComplex t2cffs[4]);


	ClassDef(TBKM02,1);


};

#endif
