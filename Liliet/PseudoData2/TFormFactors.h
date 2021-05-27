#ifndef _TFORMFACTORS_H
#define _TFORMFACTORS_H

class TFormFactors {

private:

Double_t M = 0.938272; //Mass of the proton in GeV

// Kelly's parametrization fit Parameters
Double_t a1_GEp = -0.24;
Double_t b1_GEp = 10.98;
Double_t b2_GEp = 12.82;
Double_t b3_GEp = 21.97;
Double_t a1_GMp = 0.12;
Double_t b1_GMp = 10.97;
Double_t b2_GMp = 18.86;
Double_t b3_GMp = 6.55;
// BBBA05 fit parameters


public:

	Double_t ffGE(Double_t t);
	Double_t ffGM(Double_t t);
	Double_t ffF2(Double_t t);
	Double_t ffF1(Double_t t);
	Double_t ffGA(Double_t t);
	Double_t ffGP(Double_t t);

	// Kelly parametrization
	Double_t ffGEp(Double_t t);
	Double_t ffGMp(Double_t t);
	Double_t ffF2_K(Double_t t);
	Double_t ffF1_K(Double_t t);

	ClassDef(TFormFactors,1);

};

#endif
