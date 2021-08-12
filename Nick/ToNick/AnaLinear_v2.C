#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "TVA1_UU.h"
#include "TVA1_UU.cxx"
#include "TFormFactors.cxx"
#include "TFormFactors.h"
#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "TList.h"
#include "Math/WrappedMultiTF1.h"
#include "HFitInterface.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraphMultiErrors.h"

#include "AnaLinearDef.h"

using namespace std;

//_________________________________________________________________________
void AnaLinear_v2() {

  // Read pseudo-data tree
  TFile* dataFile = TFile::Open("dvcs_xs_05-21_342_sets_15%.root");

  // Create a TTreeReader for the tree, by passing the TTree's name and the  TFile it is in.
  TTreeReader myReader("dvcs", dataFile);

  //const Long64_t NumOfSets = myReader.GetEntries();
  const Int_t NumOfSets = myReader.GetEntries();

  cout<<"Num of sets: "<<NumOfSets<<endl;

  // pseudo-data tree reader
  TTreeReaderValue<Double_t> myk(myReader, "kinematics.k");
  TTreeReaderValue<Double_t> myQQ(myReader, "kinematics.QQ");
  TTreeReaderValue<Double_t> myxB(myReader, "kinematics.xB");
  TTreeReaderValue<Double_t> myt(myReader, "kinematics.t");
  TTreeReaderArray<Double_t> phi(myReader, "phi");
  TTreeReaderArray<Double_t> F(myReader, "F");
  TTreeReaderArray<Double_t> errF(myReader, "errF");
  TTreeReaderValue<Double_t> mygdvcs(myReader, "gdvcs");
  TTreeReaderValue<Double_t> genReH(myReader, "gReH");
  TTreeReaderValue<Double_t> genReE(myReader, "gReE");

  // Graphs definitions
  TGraphErrors* gF[NumOfSets]; // Total xs vs phi
  TGraphErrors* gIntf[NumOfSets]; // BH-DVCS interf. xs vs phi
  TGraphErrors* gIntf_AB[NumOfSets]; // BH-DVCS interf. xs in A/B space

  Int_t iset = 0;
  Int_t fNSet = 1;

  // Loop through all the TTree's entries (loop over kinematic sets)
  while (myReader.Next()) {

    // behaves like an iterator
    Double_t k = *myk;
    Double_t QQ = *myQQ;
    Double_t xB = *myxB;
    Double_t t = *myt;
    Double_t gdvcs = *mygdvcs;
    Double_t gReH = *genReH;
    Double_t gReE = *genReE;

    cout << k << endl;

    // ----------------------- Fill pseudo-data xs graph ----------------------------------------
    gF[iset] = new TGraphErrors();
    gF[iset] ->SetName(Form("gF_%d",iset));
    gF[iset] ->SetTitle(Form("k = %.2f, QQ = %.2f, xB = %.2f, t = %.2f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", k, QQ, xB, t));

    for (Int_t i = 0; i < NumOfDataPoints; i++) {
      gF[iset] ->SetPoint( i, phi[i], F[i] );
      gF[iset] ->SetPointError( i, 0, errF[i] );
    }
    // ------------------------------------------------------------------------------------------

    // ----------------------- Initial fit to extract pure dvcs constant ------------------------
    TF1 * fTotalUUXS = new TF1("fTotalUUXS", TotalUUXS, 0, 360, 8);
          fTotalUUXS ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "dvcs");

    // Set Kinematics on the fit function
    fTotalUUXS ->FixParameter(0, k);
    fTotalUUXS ->FixParameter(1, QQ);
    fTotalUUXS ->FixParameter(2, xB);
    fTotalUUXS ->FixParameter(3, t);

    gF[iset] ->Fit(fTotalUUXS, "R+");

    Double_t dvcs = fTotalUUXS->GetParameter(7);
    Double_t e_dvcs = fTotalUUXS->GetParError(7);

    // Compare extracted and true dvcs values
    cout<<"dvcs(fit) = "<< dvcs << " #pm "<< e_dvcs <<", dvcs(true) = "<<gdvcs<<endl;
    // -------------------------------------------------------------------------------------------

    // ----------------------- Get BH-DVCS interference from pseudo-data -------------------------
    tva1->SetKinematics( QQ, xB, t, k );
    Double_t Gamma = tva1->GetGamma();
    Double_t F1 = ff->ffF1_K(t);
  	Double_t F2 = ff->ffF2_K(t);
    Double_t AuuI, BuuI, CuuI; // Interf. coefficients
    Double_t xs_AB, xs_AB_er; // reduced xs

    for (Int_t i = 0; i < NumOfDataPoints; i++) {

      Double_t bh = tva1 ->GetBHUU(phi[i], F1, F2);

      // -- Interference in A/B space --
      tva1->GetIUUCoefficients(phi[i], AuuI, BuuI, CuuI);
      xs_AB = - ( QQ * TMath::Abs(t) ) / ( 2. * PI * Gamma * BuuI * GeV2nb) * ( F[i] - bh - dvcs );
      // Interference error. Note: interf. error remain symmetric in A/B space
      xs_AB_er = TMath::Abs(- ( QQ * TMath::Abs(t) ) / ( 2. * PI * Gamma * BuuI * GeV2nb) * ( sqrt( errF[i] * errF[i] + e_dvcs * e_dvcs ) ) );

      cout<<"phi: "<<phi[i]<<",  F[i] = "<< F[i]<< " bh = " << bh<<", dvcs = "<< dvcs<< ", F1 = "<<F1<<", F2 = "<<F2<<", ( F[i] - bh - dvcs ) = "<<( F[i] - bh - dvcs )<<endl;
      cout<<"phi: "<<phi[i]<<", A/B = "<<AuuI/BuuI<<", Red. xs: "<<xs_AB<<" +- "<<xs_AB_er<<endl;
    }
    // -------------------------------------------------------------------------------------------

    iset++;

    if(iset == fNSet) break;
  } // end kinematics loop
}
//_________________________________________________________________________
Double_t TotalUUXS(Double_t *angle, Double_t *par)	// Total Cross Section - user defined function
{
	Double_t _phi = angle[0];
	Double_t _k = par[0];
	Double_t _QQ = par[1];
	Double_t _xB = par[2];
	Double_t _t = par[3];
	Double_t _ReH = par[4];
	Double_t _ReE = par[5];
	Double_t _ReHtilde = par[6];
	Double_t _dvcs = par[7];

	Double_t _F1 = ff->ffF1_K(_t);
	Double_t _F2 = ff->ffF2_K(_t);

	// Set QQ, xB, t and k
	tva1 ->SetKinematics( _QQ, _xB, _t, _k );

	Double_t xsbhuu	 = tva1 ->GetBHUU(_phi, _F1, _F2); // BH cross section
  Double_t xsiuu = tva1 ->GetIUU(_phi, _F1, _F2, _ReH, _ReE, _ReHtilde);

	Double_t tot_sigma_uu = xsbhuu + xsiuu + _dvcs; // Constant added to account for DVCS contribution

	return tot_sigma_uu;
}
