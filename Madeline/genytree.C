#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "/home/lily/GPDs/ANN/GenData_VA2020/BKM/TBMK.h"
#include "/home/lily/GPDs/ANN/GenData_VA2020/BKM/TBMK.cxx"
#include "/home/lily/GPDs/ANN/GenData_VA2020/BKM/TFormFactors.cxx"
#include "/home/lily/GPDs/ANN/GenData_VA2020/BKM/TFormFactors.h"
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

TBMK *bmk = new TBMK;

TFormFactors *ff = new TFormFactors;

Double_t PI = TMath::Pi();
Double_t RAD = PI / 180.;

//___________________________________________________________________________________________________
Double_t TotalUUXS_BMK(Double_t *angle, Double_t *par)	{// Total Cross Section - user defined function

	Double_t _phi = angle[0];
	Double_t _k = par[0];
	Double_t _QQ = par[1];
	Double_t _xB = par[2];
	Double_t _t = par[3];

	/* F = { H, E , Htilde, Etilde} Twist-2 Compton Form Factors*/
	TComplex _F[4] = {0};
	_F[0] = TComplex(par[4],par[8]);
	_F[1] = TComplex(par[5],par[9]);
	_F[2] = TComplex(par[6],par[10]);
	_F[3] = TComplex(par[7],par[11]);

	Double_t _F1 = ff->ffF1_K(_t);
	Double_t _F2 = ff->ffF2_K(_t);

	// Set QQ, xB, t and k
	bmk ->SetKinematics( _QQ, _xB, _t, _k );
	Double_t xsbhuu	 = bmk ->BHUU(_phi, _F1, _F2); // BH cross section
	Double_t xsintf	 = bmk ->IUU3(_phi, _F1, _F2, _F); // BH cross section
	Double_t xsdvcs	 = bmk ->DVCSUU2( _F ); // BH cross section

	Double_t tot_sigma_uu = xsbhuu + xsintf + xsdvcs; // Constant added to account for DVCS contribution

	return tot_sigma_uu;
}
//___________________________________________________________________________________________________
Double_t TotalUUXS_BMK_Fit(Double_t *angle, Double_t *par)	{// Total Cross Section - user defined function

	Double_t _phi = angle[0];
	Double_t _k = par[0];
	Double_t _QQ = par[1];
	Double_t _xB = par[2];
	Double_t _t = par[3];

	/* F = { H, E , Htilde, Etilde} Twist-2 Compton Form Factors*/
	TComplex _F[4] = {0};
	_F[0] = TComplex(par[4],0.);
	_F[1] = TComplex(par[5],0.);
	_F[2] = TComplex(par[6],0.);
	//_F[3] = TComplex(par[7],par[11]);

	Double_t xsdvcs = par[7];

	Double_t _F1 = ff->ffF1_K(_t);
	Double_t _F2 = ff->ffF2_K(_t);

	// Set QQ, xB, t and k
	bmk ->SetKinematics( _QQ, _xB, _t, _k );
	Double_t xsbhuu	 = bmk ->BHUU(_phi, _F1, _F2); // BH cross section
	Double_t xsintf	 = bmk ->IUU3(_phi, _F1, _F2, _F); // BH cross section

	Double_t tot_sigma_uu = xsbhuu + xsintf + xsdvcs ;

	return tot_sigma_uu;
}
//___________________________________________________________________________________________________
void genytree(Double_t var = 0.05) {

	Double_t PhiX;
  Int_t seed = 400;
	Double_t M = 0.938272; //Mass of the proton in GeV
	const Int_t NumOfDataPoints = 45;

	// Get kinematics from the pseudo-data generated using Liutis formulation
	TFile* inputFile = TFile::Open("/home/lily/GPDs/ANN/GenData_VA2020/Liuti/dvcs_xs_05-21_342_sets_15%.root");

	// Create a TTreeReader for the tree, by passing the TTree's name and the  TFile it is in.
	TTreeReader myReader("dvcs", inputFile);

	TTreeReaderValue<Double_t> myk(myReader, "kinematics.k");
	TTreeReaderValue<Double_t> myQQ(myReader, "kinematics.QQ");
	TTreeReaderValue<Double_t> myxB(myReader, "kinematics.xB");
	TTreeReaderValue<Double_t> myt(myReader, "kinematics.t");
	//TTreeReaderArray<Double_t> phi(myReader, "phi");

	Long64_t NumOfSets = myReader.GetEntries();

	// Functions initialization
	TF1* fgen = new TF1("fgen", TotalUUXS_BMK, 0, 360, 12);	// Generating function initialization
	TF1* ffit = new TF1("ffit", TotalUUXS_BMK_Fit, 0, 360, 8); // Fit function initialization
			 ffit ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "dvcs");


	// Pseudo-data cross section
	TGraphErrors* gGenDVCS = new TGraphErrors(NumOfDataPoints);
								gGenDVCS ->SetMarkerStyle(20);
								gGenDVCS ->SetMarkerSize(1.8);

	TGraphErrors *gSelGenDVCS[400];

	ofstream myfile; //output file
	myfile.open (Form("dvcs_bkm_xs_June2021_%.2f.csv",var));
	myfile<<"#Set,index,k,QQ,x_b,t,phi_x,F,sigmaF,varF,F1,F2,gReH,gReE,gReHTilde,ReH,e_ReH,ReE,e_ReE,ReHtilde,e_ReHtilde,gdvcs,dvcs,e_dvcs"<<endl;

	// File to save the TTree
	TFile fout(Form("dvcs_bkm_xs_June2021_%.2f.root",var),"recreate");

	TCanvas *c1;

	struct kin_t {
      Double_t k;
      Double_t QQ;
			Double_t xB;
			Double_t t;
   };
	kin_t kin;
	Double_t phi[NumOfDataPoints];
	Double_t F[NumOfDataPoints];
	Double_t errF[NumOfDataPoints];
	Double_t varF;
	Double_t gdvcs;
	Double_t dvcs, e_dvcs;
	Double_t gReH, gReE, gReHtilde, gReEtilde, ReH, ReE, ReHtilde;
	Double_t e_ReH, e_ReE, e_ReHtilde;
	Double_t gImH, gImE, gImHtilde, gImEtilde;


	TTree *t3 = new TTree("dvcs_bkm","generated dvcs");
	t3->Branch("kinematics",&kin.k,"k/D:QQ:xB:t");
	t3->Branch("phi",phi,"phi[45]/D");
	t3->Branch("F",F,"F[45]/D");
	t3->Branch("errF",errF,"errF[45]/D");
	t3->Branch("varF",&varF,"varF/D");
	t3->Branch("gReH",&gReH,"gReH/D");
	t3->Branch("gReE",&gReE,"gReE/D");
	t3->Branch("gReHtilde",&gReHtilde,"gReHtilde/D");
	t3->Branch("ReH",&ReH,"ReH/D");
	t3->Branch("ReE",&ReE,"ReE/D");
	t3->Branch("ReHtilde",&ReHtilde,"ReHtilde/D");
	t3->Branch("e_ReH",&e_ReH,"e_ReH/D");
	t3->Branch("e_ReE",&e_ReE,"e_ReE/D");
	t3->Branch("e_ReHtilde",&e_ReHtilde,"e_ReHtilde/D");
	t3->Branch("gdvcs",&gdvcs,"gdvcs/D");
	t3->Branch("dvcs",&dvcs,"dvcs/D");
	t3->Branch("e_dvcs",&e_dvcs,"e_dvcs/D");

	Int_t icand = 0, iset = 0, ig = 0;

	Long_t lNCandidates = NumOfSets;
  Long_t lOneTenthOfNCandidates = ((double)(lNCandidates) / 10. );

	cout<< "----- Kinematics -----"<<endl;

	// Loop through all the TTree's entries
	while (myReader.Next()) {

		//generate the data from random seed
		TRandom *r   = new TRandom3();
		r->SetSeed(seed+iset);

		// behaves like an iterator
		Double_t _k = *myk;
		Double_t d_QQ = *myQQ;
		Double_t d_xB = *myxB;
		Double_t d_t = *myt;

		kin.k = _k;
		kin.QQ = d_QQ;
		kin.xB = d_xB;
		kin.t = d_t;

		// Define values of CFFs and dvcs
		gReH = 5. * d_t * d_t + 2.* d_xB * d_xB;
		gReE = -1.5 * d_xB * d_xB + 4.5 * d_t;
		gReHtilde = 4.5 * d_xB - 5.5 * d_t;
		gReEtilde = 5. * d_xB - 3. * d_t;
		gImH = 0;
		gImE = 0;
		gImHtilde = 0;
		gImEtilde = 0;

		TComplex _F[4];
		_F[0] = TComplex(gReH,gImH);
		_F[1] = TComplex(gReE,gImE);
		_F[2] = TComplex(gReHtilde,gImHtilde);
		_F[3] = TComplex(gReEtilde,gImEtilde);

		fgen->FixParameter(0, _k); //k
		fgen->FixParameter(1, d_QQ); //QQ
		fgen->FixParameter(2, d_xB); //xB
		fgen->FixParameter(3, d_t); //t
		fgen->FixParameter(4, gReH);
    fgen->FixParameter(5, gReE);
    fgen->FixParameter(6, gReHtilde);
    fgen->FixParameter(7, gReEtilde);
		fgen->FixParameter(8, gImH);
    fgen->FixParameter(9, gImE);
    fgen->FixParameter(10, gImHtilde);
    fgen->FixParameter(11, gImEtilde);

		ffit->FixParameter(0, _k); //k
		ffit->FixParameter(1, d_QQ); //QQ
    ffit->FixParameter(2, d_xB); //xB
    ffit->FixParameter(3, d_t); //t

		bmk ->SetKinematics( d_QQ, d_xB, d_t, _k );
		Double_t F1 = ff->ffF1_K(d_t);
		Double_t F2 = ff->ffF2_K(d_t);

		PhiX=0;
		varF = var;
		//Generate cross section from kinematics
    for (Int_t i=0; i<NumOfDataPoints; i++) {

			PhiX = ( i * 360. / NumOfDataPoints ) + 360. / NumOfDataPoints;
			phi[i] = PhiX;
			F[i] = r->Gaus(fgen->Eval(PhiX),var*fgen->Eval(PhiX));// find cross section with 10% variation in output
			//Double_t f1 = r->Uniform(F[i],0.4);// Generate Uniform variation
			errF[i] = var*F[i];
			//Double_t f2 = 0.1*F + r->Gaus(F*.01,0.001); // A simulated physical error (with absolute and relative contributions)
			if(F[i]>0){
				// Fill pseudo data graph
				gGenDVCS ->SetPoint( i, phi[i], F[i] );
				gGenDVCS ->SetPointError( i, 0, errF[i] );
			} // end if
		}// end phi loop

		if ( F[0]>0 ) {

			// Fit raw generated pseudo-data
			gGenDVCS ->Fit("ffit","QR");

			ReH = ffit ->GetParameter(4); //ReH
			ReE = ffit ->GetParameter(5); //ReE
			ReHtilde = ffit ->GetParameter(6); //ReHtilde
			e_ReH = ffit ->GetParError(4); //ReH fit error
			e_ReE = ffit ->GetParError(5); //ReE fit error
			e_ReHtilde = ffit ->GetParError(6); //ReHtilde fit error

			dvcs = ffit ->GetParameter(7); // dvcs
			e_dvcs = ffit ->GetParError(7); // dvcs fit error

			// Select only those kinematics where the CFFs can be retrive within 20% from the generated values
			Double_t pct_change_ReH = 100. * (ReH - gReH) / gReH ;
			Double_t pct_change_ReE = 100. * (ReE - gReE) / gReE ;
			Double_t pct_change_ReHtilde = 100. * (ReHtilde - gReHtilde) / gReHtilde;

			if ( (TMath::Abs(pct_change_ReH) <= 15.) && (TMath::Abs(pct_change_ReE) <= 15.) && (TMath::Abs(pct_change_ReHtilde) <= 15.) ) {

				cout<<"QQ = "<<d_QQ<<", x_B = "<<d_xB<<", t = "<<d_t<<endl;
				cout<<"********* Good CFF extraction (<15%) *********"<<endl;
				cout<<"   ReH  	%  |	ReE  	   %   | ReHtilde     %   |"<<endl;
				cout<<ReH<<"  "<<TMath::Abs(pct_change_ReH)<<" | "	<<ReE<<"  "<<TMath::Abs(pct_change_ReE)<<" | "<<ReHtilde<<"  "<<TMath::Abs(pct_change_ReHtilde)<<" | "<<endl;

				// generated pure dvcs xs
				gdvcs = bmk ->DVCSUU2( _F ); // BH cross section

				for (Int_t i=0; i<NumOfDataPoints; i++) {
					// Print values to .csv file
					myfile<<ig+1<<","<<i<<","<<_k<<","<<d_QQ<<","<<d_xB<<","<<d_t<<","<<phi[i]<<","<<F[i]<<","<<errF[i]<<","<<var<<","<<F1<<","<<F2<<","<<gReH<<","
								<<gReE<<","<<gReHtilde<<","<<ReH<<","<<e_ReH<<","<<ReE<<","<<e_ReE<<","<<ReHtilde<<","<<e_ReHtilde<<","<<gdvcs<<","<<dvcs<<","<<e_dvcs<<endl;
				}

				gSelGenDVCS[ig] = (TGraphErrors*)gGenDVCS->Clone(Form("gSelGenDVCS_%d ", ig));

				c1 = new TCanvas(Form("c_%d",ig),"test", 552, 274, 2198, 1710);
				gSelGenDVCS[ig] ->SetTitle(Form("set %d: k = %.2f, Q^{2} = %.2f, xB = %.2f, t = %.2f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", ig+1, _k, d_QQ, d_xB, d_t));
				gSelGenDVCS[ig] ->Draw("ap");
				if (ig == 0) c1->Print("plots.pdf(","pdf");
				else c1->Print("plots.pdf","pdf");

				ig++;
				t3 ->Fill();

			}

			iset++;
			} // end if
	}	// end kinematics loop

	cout<<"No of sets: "<<iset<<endl;
	cout << "No of good kinematic sets (<15%): "<<ig<<endl;

 	c1->Print("plots.pdf]","pdf");

	//t3->Print();
	//t3->Show(0);
	fout.cd();
	t3->Write();

}
