#ifndef TBKM_H
#define TBKM_H


class TBKM {


private:

	Double_t ALP_INV = 137.0359998; // 1 / Electromagnetic Fine Structure Constant
	Double_t PI = TMath::Pi();
	Double_t RAD = PI / 180.;
	Double_t M = 0.938272; //Mass of the proton in GeV
	Double_t M2 = M*M; //Mass of the proton  squared in GeV
	Double_t GeV2nb = .389379*1000000; // Conversion from 1/GeV2 to NanoBarn

	Double_t QQ, x, t, k; // kinematics
	Double_t ee, y, xi, tmin, s, Gamma;
	Double_t  K, Ktilde_10, KD;
	Double_t P1, P2; // lepton propagators
	TComplex H, E, Htilde, Etilde; // Twist-2 CFFs

	Double_t f; // F_eff proportionality constant

	Double_t c0_BH, c1_BH, c2_BH; // BH unpolarized c coefficients (BKM02 eqs. [35, 37])
	Double_t c_dvcs; // c coefficients (BKM02 eqs. [66]) for pure DVCS
	Double_t c0_dvcs, c1_dvcs; // DVCS unpolarized Fourier harmonics (BKM02 eqs. [43, 44])
	Double_t c0_dvcs_10, c1_dvcs_10; // DVCS unpolarized Fourier harmonics (BKM10 eqs. [2.18], [2.19])
	Double_t c_dvcs_ffs, c_dvcs_effeffs, c_dvcs_efffs; // c_dvcs_unp(F,F*) coefficients (BKM10 eqs. [2.22]) for pure DVCS
	Double_t Amp2_BH, Amp2_DVCS, I; // squared amplitudes BKM02
	Double_t Amp2_DVCS_10, I_10;	// squared amplitudes BKM10
	Double_t dsigma_BH, dsigma_DVCS, dsigma_I; // 4-fold differential cross sections BKM02
	Double_t dsigma_DVCS_10, dsigma_I_10; // 4-fold differential cross sections BKM10
	Double_t DVCS10_c0fit, DVCS10_c1fit; // DVCS terms break down to c0 and c1 term
	// BKM10 interference coefficients
	Double_t C_110, C_110_V, C_110_A, C_010, C_010_V, C_010_A; // n = 0
	Double_t C_111, C_111_V, C_111_A, C_011, C_011_V, C_011_A; // n = 1
	Double_t C_112, C_112_V, C_112_A, C_012, C_012_V, C_012_A; // n = 2
	Double_t C_113, C_113_V, C_113_A; // n = 3
	Double_t A_U_I, B_U_I, C_U_I;

public:

	TBKM();  // Constructor
	~TBKM(); // Destructor

	TComplex cdstar( TComplex c, TComplex d ); // complex and complex conjugate numbers product
	void SetCFFs( TComplex *t2cffs ); // t2cffs = { H, E , Htilde, Etilde } Twist-2 Compton Form Factors
	void SetKinematics(Double_t *kine);
	void BHLeptonPropagators(Double_t *kine, Double_t phi);
	//BKM02
	Double_t BH_UU(Double_t *kine, Double_t phi, Double_t F1, Double_t F2);
	Double_t DVCS_UU_02(Double_t *kine, Double_t phi, TComplex *t2cffs, TString twist);
	Double_t I_UU_02(Double_t *kine, Double_t phi, Double_t F1, Double_t F2, TComplex *t2cffs, TString twist);
	//BKM10
	Double_t DVCS_UU_10(Double_t *kine, Double_t phi, TComplex *t2cffs, TString twist);
	Double_t Get_c0fit(Double_t *kine, TComplex *t2cffs, TString twist);
	Double_t Get_c1fit(Double_t *kine, TComplex *t2cffs, TString twist);
	Double_t I_UU_10(Double_t *kine, Double_t phi, Double_t F1, Double_t F2, TComplex *t2cffs, TString twist);
	void ABC_UU_I_10(Double_t *kine, Double_t phi, Double_t &A_U_I, Double_t &B_U_I, Double_t &C_U_I, TString twist);

	ClassDef(TBKM,1);


};

#endif

