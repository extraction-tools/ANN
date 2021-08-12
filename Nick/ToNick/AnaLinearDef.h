#ifndef ANALINEARDEF_H
#define ANALINEARDEF_H

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

TVA1_UU *tva1 = new TVA1_UU;

TFormFactors *ff = new TFormFactors;

Double_t PI = TMath::Pi();
//Double_t PI = 3.1415926535; 
Double_t RAD = PI / 180.;
Double_t M = 0.938272; //Mass of the proton in GeV
Double_t GeV2nb = .389379*1000000; // Conversion from GeV to NanoBarn

// Functions definitions
Double_t TotalUUXS(Double_t *angle, Double_t *par);
Double_t Linear_func(Double_t *angle, Double_t *par);
Double_t AuuIBuuI_ratio(Double_t *angle, Double_t *par);
Double_t CuuIBuuI_ratio(Double_t *angle, Double_t *par);
Double_t AuuICuuI_ratio(Double_t *angle, Double_t *par);

const Int_t NumOfDataPoints = 45;



#endif
