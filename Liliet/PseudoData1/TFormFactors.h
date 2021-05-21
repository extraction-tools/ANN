#ifndef _TFORMFACTORS_H
#define _TFORMFACTORS_H

class TFormFactors {

public:

	Double_t ffGE(Double_t t);
	Double_t ffGM(Double_t t);
	Double_t ffF2(Double_t t);
	Double_t ffF1(Double_t t);
	Double_t ffGA(Double_t t);
	Double_t ffGP(Double_t t);

	ClassDef(TFormFactors,1);

};

#endif
