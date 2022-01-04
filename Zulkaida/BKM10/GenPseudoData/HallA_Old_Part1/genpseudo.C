#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "TBKM.h" 
#include "TBKM.cxx" 
#include "TFormFactors.cxx"
#include "TFormFactors.h"
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

TBKM *bhdvcs = new TBKM;

TFormFactors *ff = new TFormFactors;

// user defined functions
Double_t TotalUUXS(Double_t *angle, Double_t *par);
Double_t TotalUUXS_Fit(Double_t *angle, Double_t *par);
Double_t GetC0Term(Double_t *angle, Double_t *par) ;
Double_t GetC1Term(Double_t *angle, Double_t *par); 

void genpseudo()
{
    // Read Hall B raw data
    char name[100];
    double temp[18480];
    ifstream inputfile;
    sprintf(name,"raw_data_HallA.txt");
    inputfile.open(name,ios::in);
    cout<<" Get the data from "<<name<<endl<<endl;

    for (int i = 0; i < 1080 ; i++)
     {
      inputfile >> temp[i];
     }

    Int_t seed = 400;
    Double_t M = 0.938272; //Mass of the proton in GeV
    const Int_t NumOfDataPts = 24;
    const Int_t NumOfSets = 5;
   // Kinematic settings of the JLab Hall B data
    Double_t phi_A[NumOfSets][NumOfDataPts];
    Double_t F_data[NumOfSets][NumOfDataPts];
    Double_t stat_err[NumOfSets][NumOfDataPts];
    Double_t var[NumOfSets][NumOfDataPts];
    Double_t k_A = 5.75;
    Double_t QQ_A[NumOfSets];
    Double_t xB_A[NumOfSets];
    Double_t t_A[NumOfSets];
    Double_t F1_A[NumOfSets];
    Double_t F2_A[NumOfSets]; 
   
    for (int i=0; i<NumOfSets; i++)
     {
      xB_A[i] = temp[NumOfDataPts*9*i + 2];
      QQ_A[i] = temp[NumOfDataPts*9*i + 1];
      t_A[i] = temp[NumOfDataPts*9*i + 3];
      F1_A[i] = ff->ffF1_K(t_A[i]);
      F2_A[i] = ff->ffF2_K(t_A[i]);
      for (int j=0; j<NumOfDataPts; j++)
       {
        phi_A[i][j] = temp[NumOfDataPts*9*i + 9*j + 4];
	F_data[i][j] = temp[NumOfDataPts*9*i + 9*j + 5];
        stat_err[i][j] = temp[NumOfDataPts*9*i + 9*j + 6];
	var[i][j] = stat_err[i][j]/F_data[i][j];
       }
     }

    // Generated CFFs 2D graphs vs t and xB
    TGraph2D *g2D_ReH = new TGraph2D(NumOfSets);
            g2D_ReH ->SetName("ReH");
            g2D_ReH ->SetTitle("; t [GeV^{2}]; x_{B};ReH");
            g2D_ReH ->SetMarkerSize(2);
            g2D_ReH ->SetMarkerStyle(22);
    		g2D_ReH ->SetLineColor(1);
    		g2D_ReH ->SetLineWidth(2);
    TGraph2D *g2D_ReE = new TGraph2D(NumOfSets);
            g2D_ReE ->SetName("ReE");
            g2D_ReE ->SetTitle("; t [GeV^{2}]; x_{B};ReE");
            g2D_ReE ->SetMarkerSize(2);
            g2D_ReE ->SetMarkerStyle(22);
            g2D_ReE ->SetLineColor(1);
            g2D_ReE ->SetLineWidth(2);
    TGraph2D *g2D_ReHtilde = new TGraph2D(NumOfSets);
            g2D_ReHtilde ->SetName("ReHtilde");
            g2D_ReHtilde ->SetTitle("; t [GeV^{2}]; x_{B};ReHtilde");
            g2D_ReHtilde ->SetMarkerSize(2);
            g2D_ReHtilde ->SetMarkerStyle(22);
    		g2D_ReHtilde ->SetLineColor(1);
    		g2D_ReHtilde ->SetLineWidth(2);

    // Generating function initialization
	TF1* fgen = new TF1("fgen", TotalUUXS, 0, 360, 12);
	TF1* getC0_dvcs = new TF1("getC0_dvcs", GetC0Term, 0, 360, 12);
	TF1* getC1_dvcs = new TF1("getC1_dvcs", GetC1Term, 0, 360, 12);
	// Fit function initialization
 	TF1* ffit = new TF1("ffit", TotalUUXS_Fit, 0, 360, 9);
 			 ffit ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "c0fit", "c1fit");

    // Pseudo-data cross section
    TGraphErrors* gGenDVCS = new TGraphErrors(NumOfDataPts);
    							gGenDVCS ->SetMarkerStyle(20);
    							gGenDVCS ->SetMarkerSize(1.8);
    TGraphErrors *gSelGenDVCS[400];
    TCanvas *c_F;

    ofstream myfile; //output file
	myfile.open ("pseudo_BKM10_hallA_t2_nosmear.csv");
	myfile<<"#Set,index,k,QQ,x_b,t,phi_x,F,sigmaF,varF,F1,F2,gReH,gReE,gReHTilde,gc0fit,gc1fit"<<endl;

	// File to save the TTree
	TFile fout("pseudo_dvcs_xs_hallA_genkin.root","recreate");

    struct kin_t {
      Double_t k;
      Double_t QQ;
    		Double_t xB;
    		Double_t t;
    };
	kin_t kin;
	Double_t phi[NumOfDataPts];
	Double_t F[NumOfDataPts]; Double_t errF[NumOfDataPts];
	Double_t varF;
	Double_t gdvcs, dvcs, e_dvcs, c0fit, e_c0fit, c1fit, e_c1fit;
	Double_t gReH, gReE, gReHtilde, gReEtilde, gC0, gC1, ReH, ReE, ReHtilde;
	Double_t e_ReH, e_ReE, e_ReHtilde;
	Double_t gImH, gImE, gImHtilde, gImEtilde;

    TTree *t2 = new TTree("dvcs","generated dvcs");
    t2->Branch("kinematics",&kin.k,"k/D:QQ:xB:t");
    t2->Branch("phi",phi,"phi[45]/D");
    t2->Branch("F",F,"F[45]/D");
    t2->Branch("errF",errF,"errF[45]/D");
    t2->Branch("varF",&varF,"varF/D");
    t2->Branch("gReH",&gReH,"gReH/D");
    t2->Branch("gReE",&gReE,"gReE/D");
    t2->Branch("gReHtilde",&gReHtilde,"gReHtilde/D");
    t2->Branch("ReH",&ReH,"ReH/D");
    t2->Branch("ReE",&ReE,"ReE/D");
    t2->Branch("ReHtilde",&ReHtilde,"ReHtilde/D");
    t2->Branch("e_ReH",&e_ReH,"e_ReH/D");
    t2->Branch("e_ReE",&e_ReE,"e_ReE/D");
    t2->Branch("e_ReHtilde",&e_ReHtilde,"e_ReHtilde/D");
   // t2->Branch("gdvcs",&gdvcs,"gdvcs/D");
   // t2->Branch("dvcs",&dvcs,"dvcs/D");
   // t2->Branch("e_dvcs",&e_dvcs,"e_dvcs/D");


    TRandom *r   = new TRandom3();

    Int_t ig = 0;

    for (Int_t iset = 0; iset < NumOfSets; iset++) {
        //generate the data from random seed
        r->SetSeed(seed+iset);

        // Set QQ, xB, t and k
        kin.k = k_A;
        kin.QQ = QQ_A[iset];
        kin.xB = xB_A[iset];
        kin.t = t_A[iset];

        // Define values of CFFs
       
	//gReH = 5.* t_A[iset] * t_A[iset] + 2.* xB_A[iset] * xB_A[iset];
        //gReE = 4.5 * t_A[iset] - 1.5 * xB_A[iset] * xB_A[iset];
        //gReHtilde =  -5.5* t_A[iset] + 4.5 *  xB_A[iset];
	gReH = 100.* t_A[iset] * t_A[iset] + 2.* xB_A[iset] * xB_A[iset];
        gReE = 4.5 * t_A[iset] - 100 * xB_A[iset] * xB_A[iset];
        gReHtilde =  -5.5* t_A[iset] + 4.5 *  xB_A[iset];
        gReEtilde = -3. * t_A[iset] + 5. * xB_A[iset];
        gImH = 0;
        gImE = 0;
        gImHtilde = 0;
        gImEtilde = 0;

        // Fix gen, getC0 and getC1 dvcs function parameters
        fgen->FixParameter(0, k_A); //k
        fgen->FixParameter(1, QQ_A[iset]); //QQ
        fgen->FixParameter(2, xB_A[iset]); //xB
        fgen->FixParameter(3, t_A[iset]); //t
        fgen->FixParameter(4, gReH);
        fgen->FixParameter(5, gReE);
        fgen->FixParameter(6, gReHtilde);
        fgen->FixParameter(7, gReEtilde);
        fgen->FixParameter(8, gImH);
        fgen->FixParameter(9, gImE);
        fgen->FixParameter(10, gImHtilde);
        fgen->FixParameter(11, gImEtilde);

	getC0_dvcs->FixParameter(0, k_A); //k
        getC0_dvcs->FixParameter(1, QQ_A[iset]); //QQ
        getC0_dvcs->FixParameter(2, xB_A[iset]); //xB
        getC0_dvcs->FixParameter(3, t_A[iset]); //t
        getC0_dvcs->FixParameter(4, gReH);
        getC0_dvcs->FixParameter(5, gReE);
        getC0_dvcs->FixParameter(6, gReHtilde);
        getC0_dvcs->FixParameter(7, gReEtilde);
        getC0_dvcs->FixParameter(8, gImH);
        getC0_dvcs->FixParameter(9, gImE);
        getC0_dvcs->FixParameter(10, gImHtilde);
        getC0_dvcs->FixParameter(11, gImEtilde);

	getC1_dvcs->FixParameter(0, k_A); //k
        getC1_dvcs->FixParameter(1, QQ_A[iset]); //QQ
        getC1_dvcs->FixParameter(2, xB_A[iset]); //xB
        getC1_dvcs->FixParameter(3, t_A[iset]); //t
        getC1_dvcs->FixParameter(4, gReH);
        getC1_dvcs->FixParameter(5, gReE);
        getC1_dvcs->FixParameter(6, gReHtilde);
        getC1_dvcs->FixParameter(7, gReEtilde);
        getC1_dvcs->FixParameter(8, gImH);
        getC1_dvcs->FixParameter(9, gImE);
        getC1_dvcs->FixParameter(10, gImHtilde);
        getC1_dvcs->FixParameter(11, gImEtilde);
        // Fix fit function parameters
        ffit->FixParameter(0, k_A); //k
        ffit->FixParameter(1, QQ_A[iset]); //QQ
        ffit->FixParameter(2, xB_A[iset]); //xB
        ffit->FixParameter(3, t_A[iset]); //t

        // Fill 2D graphs for ReH, ReE and ReHtilde
        g2D_ReH->SetPoint(iset,t_A[iset], xB_A[iset], gReH );
        g2D_ReE->SetPoint(iset,t_A[iset], xB_A[iset], gReE );
        g2D_ReHtilde->SetPoint(iset,t_A[iset], xB_A[iset], gReHtilde );

        //Generate cross section from kinematics
        for (Int_t i=0; i<NumOfDataPts; i++) {

            phi[i] = phi_A[iset][i];
            //F[i] = r->Gaus(fgen->Eval(phi[i]),var[iset][i]*fgen->Eval(phi[i]));// find cross section with variation in output based on the stat err of data
	    F[i] = fgen->Eval(phi[i]);// special case for No smearing. comment out this line for normal smearing and use the line above
            errF[i] = var[iset][i]*F[i];
	    gC0 = getC0_dvcs->Eval(phi[i]);
	    gC1 = getC1_dvcs->Eval(phi[i]);

            // Fill pseudo data graph
          if(F_data[iset][i] > 0) {
            gGenDVCS ->SetPoint( i, phi[i], F[i] );
            gGenDVCS ->SetPointError( i, 0, errF[i] );
          }
        }// end phi loop

        if ( F[0] < 0 ) { cout<<" \"Not Valid Kinematic Set\" --> Skipped"<<endl; continue; }

        // Fit raw generated pseudo-data
        gGenDVCS ->Fit("ffit","QR");

        // Get fit parameters
        ReH = ffit ->GetParameter(4); //ReH
        ReE = ffit ->GetParameter(5); //ReE
        ReHtilde = ffit ->GetParameter(6); //ReHtilde
        e_ReH = ffit ->GetParError(4); //ReH fit error
        e_ReE = ffit ->GetParError(5); //ReE fit error
        e_ReHtilde = ffit ->GetParError(6); //ReHtilde fit error
        c0fit = ffit ->GetParameter(7); //c0fit
        e_c0fit = ffit ->GetParError(7); //c0fit error
	c1fit = ffit ->GetParameter(8); //c1fit   
        e_c1fit = ffit ->GetParError(8); //c1fit error

        // Get percent change
        Double_t pct_change_ReH = 100. * (ReH - gReH) / gReH ;
        Double_t pct_change_ReE = 100. * (ReE - gReE) / gReE ;
        Double_t pct_change_ReHtilde = 100. * (ReHtilde - gReHtilde) / gReHtilde;

        cout<<"set: "<<iset+1<<" QQ = "<<QQ_A[iset]<<", x_B = "<<xB_A[iset]<<", t = "<<t_A[iset]<<endl;
        cout<<"********* CFF extraction *********"<<endl;
        cout<<"   ReH  	%       |	ReE  	   %        | ReHtilde     %        |"<<endl;
        cout<<ReH<<" ( "<<gReH<<" )  "<<TMath::Abs(pct_change_ReH)<<" | "	<<ReE<<" ( "<<gReE<<" ) "<<TMath::Abs(pct_change_ReE)<<" | "<<ReHtilde<<" ( "<<gReHtilde<<" ) "<<TMath::Abs(pct_change_ReHtilde)<<" | "<<endl;
        cout<<endl;
        cout<<endl;
/*
        // Get gen dvcs xs
        bhdvcs ->SetKinematics( QQ_A[iset], xB_A[iset], t_A[iset], k_A );
        gdvcs = bhdvcs ->GetDVCSUUxs( gReH, gReE, gReHtilde, gReEtilde, gImH, gImE, gImHtilde, gImEtilde);
*/
        // Save the xs graphs for the kinematics settings
        gSelGenDVCS[ig] = (TGraphErrors*)gGenDVCS->Clone(Form("gSelGenDVCS_%d ", ig));
        gSelGenDVCS[ig] ->SetTitle(Form("set %d: k = %.2f, Q^{2} = %.2f, xB = %.2f, t = %.2f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", ig+1, k_A, QQ_A[iset], xB_A[iset], t_A[iset]));
        // Print results to a .csv file
        for (Int_t i=0; i<NumOfDataPts; i++) {
	   if(F_data[iset][i] > 0) {
             myfile<<ig+1<<","<<i<<","<<k_A<<","<<QQ_A[iset]<<","<<xB_A[iset]<<","<<t_A[iset]<<","<<phi[i]<<","<<F[i]<<","<<errF[i]<<","<<errF[i]/F[i]<<","<<F1_A[iset]<<","<<F2_A[iset]<<","<<gReH<<","
                <<gReE<<","<<gReHtilde<<","<<gC0<<","<<gC1<<endl;
	     }
	   else {
	     myfile<<ig+1<<","<<i<<","<<k_A<<","<<QQ_A[iset]<<","<<xB_A[iset]<<","<<t_A[iset]<<","<<phi[i]<<","<<0<<","<<0<<","<<0<<","<<0<<","<<0<<","<<0<<","
                <<0<<","<<0<<endl; 
	    }
        }

        // Draw F(cross section) distributions
        c_F = new TCanvas(Form("c_F_%d",ig),"test", 552, 274, 2198, 1710);
        gSelGenDVCS[ig] ->SetTitle(Form("set %d: k = %.2f, Q^{2} = %.2f, xB = %.2f, t = %.2f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", ig+1, k_A, QQ_A[iset], xB_A[iset], t_A[iset]));
        gSelGenDVCS[ig] ->Draw("ap");

        if (ig == 0) c_F->Print("plots_5pct_2.pdf(","pdf");
        else c_F->Print("plots_5pct_2.pdf","pdf");

        ig++;
        t2 ->Fill();

    }// end kine loop

    for(Int_t i = 0; i<NumOfSets; i=i+5){
        c_F = new TCanvas(Form("c_F_%d",i),"test", 552, 274, 2198, 1710);
        c_F ->Divide(3,2);
        c_F ->cd(1);
        gSelGenDVCS[i] ->Draw("ap");
        c_F ->cd(2);
        gSelGenDVCS[i+1] ->Draw("ap");
        c_F ->cd(3);
        gSelGenDVCS[i+2] ->Draw("ap");
        c_F ->cd(4);
        gSelGenDVCS[i+3] ->Draw("ap");
        c_F ->cd(5);
        gSelGenDVCS[i+4] ->Draw("ap");

        if (i == 0) c_F->Print("plots_5pct_2.pdf(","pdf");
        else c_F->Print("plots_5pct_2.pdf","pdf");
    }

    c_F->Print("plots_5pct_2.pdf]","pdf");

    TCanvas *c_cffs = new TCanvas("c_cffs", "c_cffs",1134,613,2706,931);
    c_cffs->Divide(3,1);
    c_cffs->cd(1);
    g2D_ReH->Draw("TRI1 P0");
    c_cffs->cd(2);
    g2D_ReE->Draw("TRI1 P0");
    c_cffs->cd(3);
    g2D_ReHtilde->Draw("TRI1 P0");

    fout.cd();
    t2->Write();

}
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
	Double_t _ReEtilde = par[7];
	Double_t _ImH = par[8];
	Double_t _ImE = par[9];
	Double_t _ImHtilde = par[10];
	Double_t _ImEtilde = par[11];

	Double_t _F1 = ff->ffF1_K(_t);
	Double_t _F2 = ff->ffF2_K(_t);

	Double_t AUUI, BUUI, CUUI;

	// Set QQ, xB, t and k
        Double_t kinset[4] = {_QQ, _xB, _t, _k};
	bhdvcs ->SetKinematics(kinset);
	Double_t xsbhuu	 = bhdvcs ->BH_UU(kinset, _phi, _F1, _F2); // BH cross section
	
	TComplex _H(_ReH, _ImH);
	TComplex _E(_ReE, _ImE);
	TComplex _HTilde(_ReHtilde, _ImHtilde);
	TComplex _ETilde(_ReEtilde, _ImEtilde);
	TComplex CFFs[4] = {_H, _E, _HTilde, _ETilde};
        Double_t xsiuu = bhdvcs ->I_UU_10(kinset, _phi, _F1, _F2, CFFs, "t2"); // Interference
	Double_t xsdvcs = bhdvcs ->DVCS_UU_10(kinset, _phi, CFFs, "t2"); // dvcs

	Double_t tot_sigma_uu = xsbhuu + xsiuu + xsdvcs; 

	return tot_sigma_uu;
}

Double_t GetC0Term(Double_t *angle, Double_t *par)      // Total Cross Section - user defined function
{
        Double_t _phi = angle[0];
        Double_t _k = par[0];
        Double_t _QQ = par[1];
        Double_t _xB = par[2];
        Double_t _t = par[3];
        Double_t _ReH = par[4];
        Double_t _ReE = par[5];
        Double_t _ReHtilde = par[6];
        Double_t _ReEtilde = par[7];
        Double_t _ImH = par[8];
        Double_t _ImE = par[9];
        Double_t _ImHtilde = par[10];
        Double_t _ImEtilde = par[11];
        Double_t _F1 = ff->ffF1_K(_t);
        Double_t _F2 = ff->ffF2_K(_t);
        Double_t AUUI, BUUI, CUUI;
        // Set QQ, xB, t and k
        Double_t kinset[4] = {_QQ, _xB, _t, _k};
        bhdvcs ->SetKinematics(kinset);

        TComplex _H(_ReH, _ImH);
        TComplex _E(_ReE, _ImE);
        TComplex _HTilde(_ReHtilde, _ImHtilde);
        TComplex _ETilde(_ReEtilde, _ImEtilde);
        TComplex CFFs[4] = {_H, _E, _HTilde, _ETilde};

        Double_t term_c0 = bhdvcs ->Get_c0fit(kinset, CFFs, "t2"); 
 
        return term_c0;
}

Double_t GetC1Term(Double_t *angle, Double_t *par)      // Total Cross Section - user defined function
{
        Double_t _phi = angle[0];
        Double_t _k = par[0];
        Double_t _QQ = par[1];
        Double_t _xB = par[2];
        Double_t _t = par[3];
        Double_t _ReH = par[4];
        Double_t _ReE = par[5];
        Double_t _ReHtilde = par[6];
        Double_t _ReEtilde = par[7];
        Double_t _ImH = par[8];
        Double_t _ImE = par[9];
        Double_t _ImHtilde = par[10];
        Double_t _ImEtilde = par[11];
        Double_t _F1 = ff->ffF1_K(_t);
        Double_t _F2 = ff->ffF2_K(_t);
        Double_t AUUI, BUUI, CUUI;
        // Set QQ, xB, t and k
        Double_t kinset[4] = {_QQ, _xB, _t, _k};
        bhdvcs ->SetKinematics(kinset);
        TComplex _H(_ReH, _ImH);
        TComplex _E(_ReE, _ImE);
        TComplex _HTilde(_ReHtilde, _ImHtilde);
        TComplex _ETilde(_ReEtilde, _ImEtilde);
        TComplex CFFs[4] = {_H, _E, _HTilde, _ETilde};
        Double_t term_c1 = bhdvcs ->Get_c1fit(kinset, CFFs, "t2"); 
          
        return term_c1;
}

//____________________________________________________________________________________________________
Double_t TotalUUXS_Fit(Double_t *angle, Double_t *par)	// Total Cross Section - user defined function
{
	Double_t PI = TMath::Pi();
        Double_t RAD = PI / 180.;

	Double_t _phi = angle[0];
	Double_t _k = par[0];
	Double_t _QQ = par[1];
	Double_t _xB = par[2];
	Double_t _t = par[3];
	Double_t _ReH = par[4];
	Double_t _ReE = par[5];
	Double_t _ReHtilde = par[6];
	Double_t _c0fit = par[7];
	Double_t _c1fit = par[8];

	Double_t _F1 = ff->ffF1_K(_t);
	Double_t _F2 = ff->ffF2_K(_t);

	Double_t AUUI, BUUI, CUUI;

	// Set QQ, xB, t and k
	Double_t kinset[4] = {_QQ, _xB, _t, _k};
	bhdvcs ->SetKinematics(kinset);
        Double_t xsbhuu  = bhdvcs ->BH_UU(kinset, _phi, _F1, _F2); // BH cross section

	TComplex _H(_ReH, 0.);
        TComplex _E(_ReE, 0.);
        TComplex _HTilde(_ReHtilde, 0.);
        TComplex _ETilde(0., 0.);
        TComplex CFFs[4] = {_H, _E, _HTilde, _ETilde};
	Double_t xsiuu = bhdvcs ->I_UU_10(kinset, _phi, _F1, _F2, CFFs, "t2"); // Interference

	//for dvcs term:
	Double_t M = 0.938272; //Mass of the proton in GeV
        Double_t M2 = M*M; //Mass of the proton  squared in GeV
	Double_t ALP_INV = 137.0359998;
	Double_t GeV2nb = .389379*1000000; 

	Double_t ee = 4. * M2 * _xB * _xB / _QQ; // epsilon squared
        Double_t y = sqrt(_QQ) / ( sqrt(ee) * _k );  // lepton energy fraction
        Double_t Gamma = _xB * y * y / ALP_INV / ALP_INV / ALP_INV / PI / 8. / _QQ / _QQ / sqrt( 1. + ee );         

	Double_t tot_sigma_uu = xsbhuu + xsiuu + 1. / ( y * y * _QQ ) * Gamma * GeV2nb * ( _c0fit + _c1fit * cos( PI - (_phi * RAD) ) ); 

	return tot_sigma_uu;
}

