#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "/home/lily/GPDs/BMK/BMK_Liuti_Check/TBHDVCS.h"
#include "/home/lily/GPDs/BMK/BMK_Liuti_Check/TBHDVCS.cxx"
#include "/home/lily/GPDs/BMK/BMK_Liuti_Check/TFormFactors.cxx"
#include "/home/lily/GPDs/BMK/BMK_Liuti_Check/TFormFactors.h"
#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "TList.h"
#include "Math/WrappedMultiTF1.h"
#include "HFitInterface.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TCanvas.h"
#include "TStyle.h"

using namespace std;

TBHDVCS *bhdvcs = new TBHDVCS;

TFormFactors *ff = new TFormFactors;

//____________________________________________________________________________________________________
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

	Double_t AUUI, BUUI, CUUI;

	// Set QQ, xB, t and k
	bhdvcs ->SetKinematics( _QQ, _xB, _t, _k );

	Double_t xsbhuu	 = bhdvcs ->GetBHUUxs(_phi, _F1, _F2); // BH cross section
  Double_t xsiuu = bhdvcs ->GetIUUxs(_phi, _F1, _F2, _ReH, _ReE, _ReHtilde, AUUI, BUUI, CUUI);

	Double_t tot_sigma_uu = xsbhuu + xsiuu + _dvcs; // Constant added to account for DVCS contribution

	return tot_sigma_uu;
}
//___________________________________________________________________________________________________
void genytree(Double_t var = 0.05) {

	Double_t _k ; Double_t PhiX;
  Int_t seed = 400;

	// JLab Hall A kinematics
	const Int_t NumOfSets = 20;
	//Double_t k = 5.75;
	Double_t _QQ[NumOfSets] = { 1.82, 1.933, 1.964, 1.986, 1.999, 2.218, 2.318, 2.348, 2.36, 2.375, 2.012, 2.054, 2.074, 2.084, 2.091, 2.161, 2.19, 2.194, 2.191, 2.193};
	Double_t _xB[NumOfSets] = { 0.343, 0.368, 0.375, 0.379, 0.381, 0.345, 0.363, 0.368, 0.371, 0.373, 0.378, 0.392, 0.398, 0.4, 0.401, 0.336, 0.342, 0.343, 0.342, 0.342};
	Double_t _t[NumOfSets] = { -0.172, -0.232, -0.278, -0.323, -0.371, -0.176, -0.232, -0.279, -0.325, -0.372, -0.192, -0.233, -0.279, -0.324, -0.371, -0.171, -0.231, -0.278, -0.324, -0.371};

	// Fit Function initialization
	TF1* fl = new TF1("fl", TotalUUXS, 0, 360, 8);
			 fl ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "DVCSXS");

	ofstream myfile; //output file
	//myfile.open (Form("DVCS_xs_%.0f.csv", 100*_varF));
	myfile.open (Form("raw_dvcs_xs_11-19-20_%.2f.csv",var));
	myfile<<"#Set,index,k,QQ,x_b,t,phi_x,F,sigmaF,varF,F1,F2,ReH,ReE,ReHTilde,dvcs"<<endl;

	// File to save the TTree
	TFile fout(Form("raw_dvcs_xs_11-19-20_%.2f.root",var),"recreate");

	struct kin_t {
      Double_t k;
      Double_t QQ;
			Double_t xB;
			Double_t t;
   };
	kin_t kin;
	Double_t phi[36];
	Double_t F[36]; Double_t errF[36];
	Double_t varF;
	Double_t dvcs;
	Double_t ReH, ReE, ReHtilde;

	TTree *t3 = new TTree("dvcs","generated dvcs");
	t3->Branch("kinematics",&kin.k,"k/D:QQ:xB:t");
	t3->Branch("phi",phi,"phi[36]/D");
	t3->Branch("F",F,"F[36]/D");
	t3->Branch("errF",errF,"errF[36]/D");
	t3->Branch("varF",&varF,"varF/D");
	t3->Branch("dvcs",&dvcs,"dvcs/D");
	t3->Branch("ReH",&ReH,"ReH/D");
	t3->Branch("ReE",&ReE,"ReE/D");
	t3->Branch("ReHtilde",&ReHtilde,"ReHtilde/D");

	for (Int_t m=0; m<6; m++) {
		_k=2.75+m;

		for ( Int_t n = 0; n < NumOfSets; n++ ) { // loop on kinematic settings
			//generate the data from random seed
	    TRandom *s   = new TRandom3();
	    s->SetSeed(seed-n);
	    //kinematic variation based on Hall A physics
	    Double_t d_QQ =  s->Gaus(_QQ[n],0.1*_QQ[n]); //using 10% variation
			Double_t d_xB =  s->Gaus(_xB[n],0.1*_xB[n]); //using 10% variation

			for ( Int_t nx = 0; nx < NumOfSets; nx++ ) {

				TRandom *r   = new TRandom3();
				TRandom *sx   = new TRandom3();
				r->SetSeed(seed+n+nx);
				sx->SetSeed(seed-n-nx);

				Double_t d_t  =  sx->Gaus(_t[nx],0.1*_t[nx]); //using 10% variation



				if ( -d_t/d_QQ > 0.2 ) continue; // DVCS kinematic limit

				kin.k = _k;
				kin.QQ = d_QQ;
				kin.xB = d_xB;
				kin.t = d_t;

			 	fl->FixParameter(0, _k); //k
				fl->FixParameter(1, d_QQ); //QQ
		    fl->FixParameter(2, d_xB); //xB
		    fl->FixParameter(3, d_t); //t

				// Define values of CFFs and dvcs
				// ReH = ff->ffF1(d_t);
				// ReE = ff->ffF2(d_t);
				// ReHtilde = ff->ffGA(d_t);
				ReH = 90. * d_t * d_t;
				ReE = - 90. * d_t * d_t - 40.;
				ReHtilde = 50. * d_t * d_t;


				dvcs = 0.0122881;

				fl->FixParameter(4, ReH); //ReH
		    fl->FixParameter(5, ReE); //ReE
		    fl->FixParameter(6, ReHtilde); //ReHtilde
		    fl->FixParameter(7, dvcs); //DVCSXS

				PhiX=0; //Generate cross section from kinematics
				varF = var;
		    for (Int_t i=0; i<36; i++) {
		       //PhiX = i*10+r->Gaus(5,3);// choose the phi
					 PhiX = i * 10;
					 phi[i] = PhiX;
		       F[i] = r->Gaus(fl->Eval(PhiX),var*fl->Eval(PhiX));// find cross section with 10% variation in output
		       //Double_t f1 = r->Uniform(F[i],0.4);// Generate Uniform variation
					 errF[i] = var*F[i];
		       //Double_t f2 = 0.1*F + r->Gaus(F*.01,0.001); // A simulated physical error (with absolute and relative contributions)
		       //if(F>0){myfile <<k<<" "<<d_QQ<<","<<d_xB<<","<<d_t<<","<<PhiX<<","<<F<<","<<TMath::Abs(f2)<<endl;}
		       if(F[i]>0)myfile<<n<<","<<i<<","<<_k<<","<<_QQ[n]<<","<<_xB[n]<<","<<_t[n]<<","<<PhiX<<","<<F<<","<<errF[i]<<","<<var<<","<<ff->ffF1_K(_t[n])<<","<<ff->ffF2_K(_t[n])<<","<<fl->GetParameter(4)<<","
					 							<<fl->GetParameter(5)<<","<<fl->GetParameter(6)<<","<<fl->GetParameter(7)<<endl;
				}// end phi loop

			if ( F[0]>0 ) t3 ->Fill();

		}//end xB loop

		} // end loop on settings
	}	// end k loop

	t3->Print();
	t3->Show(1);
	fout.cd();
	t3->Write();

}
