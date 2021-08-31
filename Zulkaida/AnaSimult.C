#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "TVA1_UU.h"
#include "TVA1_UU.cxx"
#include "/home/lily/GPDs/BMK/BMK_Liuti_Check/TFormFactors.cxx"
#include "/home/lily/GPDs/BMK/BMK_Liuti_Check/TFormFactors.h"
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
void AnaSimult() {

  // Read HallA datafile
  TFile* file = TFile::Open("/home/lily/GPDs/JLabHallAData/XSDataJLabHallA.root", "READ");

  // Graphs definitions
	TGraphErrors* gF[NumOfSets]; // Total xs vs phi
  TGraphErrors* gIntf[NumOfSets]; // BH-DVCS interf. xs vs phi
  TGraphErrors* gIntf_AB[NumOfSets]; // BH-DVCS interf. xs in A/B space
  TGraphErrors* gMapped_Control[NumOfSets]; // Reduced xs vs A/B (all points)
  TGraphErrors* gMapped_Control_w[NumOfSets]; // Reduced xs vs A/B (combining 2 points wth same A/B)

  // Graphs initializations
  for (Int_t imethod = 0; imethod < 3; imethod++){
    gReH[imethod] = new TGraphErrors();
    gReH[imethod] ->SetName(Form("%s", fitmethod[imethod].Data()));
    gReH[imethod] ->SetMarkerColor(col[imethod]);
    gReH[imethod] ->SetLineColor(col[imethod]);
    gReH[imethod] ->SetMarkerStyle(fMarker[imethod]);
    gReH[imethod] ->SetMarkerSize(1.3);

    gReE[imethod] = new TGraphErrors();
    gReE[imethod] ->SetName(Form("%s", fitmethod[imethod].Data()));
    gReE[imethod] ->SetMarkerColor(col[imethod]);
    gReE[imethod] ->SetLineColor(col[imethod]);
    gReE[imethod] ->SetMarkerStyle(fMarker[imethod]);
    gReE[imethod] ->SetMarkerSize(1.3);

    gReHt[imethod] = new TGraphErrors();
    gReHt[imethod] ->SetName(Form("%s", fitmethod[imethod].Data()));
    gReHt[imethod] ->SetMarkerColor(col[imethod]);
    gReHt[imethod] ->SetLineColor(col[imethod]);
    gReHt[imethod] ->SetMarkerStyle(fMarker[imethod]);
    gReHt[imethod] ->SetMarkerSize(1.3);

    gReH_kin3[imethod] = new TGraphErrors();
    gReH_kin3[imethod] ->SetName(Form("%s", fitmethod[imethod].Data()));
    gReH_kin3[imethod] ->SetMarkerColor(col[imethod]);
    gReH_kin3[imethod] ->SetLineColor(col[imethod]);
    gReH_kin3[imethod] ->SetMarkerStyle(fMarker[imethod]);
    gReH_kin3[imethod] ->SetMarkerSize(1.3);

    gReE_kin3[imethod] = new TGraphErrors();
    gReE_kin3[imethod] ->SetName(Form("%s", fitmethod[imethod].Data()));
    gReE_kin3[imethod] ->SetMarkerColor(col[imethod]);
    gReE_kin3[imethod] ->SetLineColor(col[imethod]);
    gReE_kin3[imethod] ->SetMarkerStyle(fMarker[imethod]);
    gReE_kin3[imethod] ->SetMarkerSize(1.3);

    gReHt_kin3[imethod] = new TGraphErrors();
    gReHt_kin3[imethod] ->SetName(Form("%s", fitmethod[imethod].Data()));
    gReHt_kin3[imethod] ->SetMarkerColor(col[imethod]);
    gReHt_kin3[imethod] ->SetLineColor(col[imethod]);
    gReHt_kin3[imethod] ->SetMarkerStyle(fMarker[imethod]);
    gReHt_kin3[imethod] ->SetMarkerSize(1.3);
  }
  TGraphErrors* gReH_ref_KM15 = new TGraphErrors("/home/lily/GPDs/LiutiNew/Simu_updated/KM15modelfit.csv", "%lg %lg %lg %lg",",");
                gReH_ref_KM15 ->SetName("KM15");
								gReH_ref_KM15 ->SetMarkerColor(1);
								gReH_ref_KM15 ->SetLineColor(1);
								gReH_ref_KM15 ->SetMarkerStyle(21);
								gReH_ref_KM15 ->SetMarkerSize(1.3);


  TGraph *gABvsPhi[NumOfSets]; // A/B vs phi

  char graphname[80];

  Int_t fNSet = NumOfSets;
  Int_t nkin3 = 0;

  for (Int_t iset = 0; iset < NumOfSets; iset++)	{ // loop on experimental kinematic settings

    if ( iset == 5 ) cout<< "**************** Kin3 Starts **********************"<<endl;
    if ( iset == 10 ) cout<< "**************** Kin3 Ends **********************"<<endl;

    // ------------------------ Get JLab data graph from the .root file ------------------------
    sprintf(graphname,"XSUU_Q2_%.3f", QQ[iset]);
    gF[iset] = (TGraphErrors*)file->FindObjectAny(graphname);
    gF[iset] ->SetTitle(Form("Q^{2} = %.3f _ xB = %.3f  _ t = %.3f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", QQ[iset], xB[iset], t[iset]));
    gF[iset] ->Sort();

    Int_t NumOfPts = gF[iset]->GetN();

    cout<<"Number of Points: "<<NumOfPts<<endl;
    // ------------------------------------------------------------------------------------------

    // ----------------------- Initial fit to extract pure dvcs constant ------------------------
    TF1 * fTotalUUXS = new TF1("fTotalUUXS", TotalUUXS, 0, 360, 8);
          fTotalUUXS ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "dvcs");
          fTotalUUXS ->SetLineColor(1);

    // Set Kinematics on the fit function
    fTotalUUXS ->FixParameter(0, k);
    fTotalUUXS ->FixParameter(1, QQ[iset]);
    fTotalUUXS ->FixParameter(2, xB[iset]);
    fTotalUUXS ->FixParameter(3, t[iset]);

    gF[iset] ->Fit(fTotalUUXS, "R+");

    Double_t dvcs = fTotalUUXS->GetParameter(7);
    Double_t e_dvcs = fTotalUUXS->GetParError(7);
    // -------------------------------------------------------------------------------------------

    // ----------------------- Get BH-DVCS interference from pseudo-data -------------------------
    tva1->SetKinematics( QQ[iset], xB[iset], t[iset], k );
    Gamma = tva1->GetGamma();
    F1 = ff->ffF1_K(t[iset]);
  	F2 = ff->ffF2_K(t[iset]);

    Double_t AB_left_limit = 0, AB_right_limit = -100;

    for (Int_t i = 0; i < NumOfPts; i++) {

      gF[iset] ->GetPoint(i, phi, F);   // raw experimental data
      errF = gF[iset] ->GetErrorY(i); 	// raw experimental error

      bh = tva1 ->GetBHUU(phi, F1, F2);
      tva1->GetIUUCoefficients(phi, AuuI, BuuI, CuuI);

      fPhi[i] = phi;
      fAB[i] = AuuI/BuuI;

      // -- Interference in phi space --
      fInt[i] = F - bh - dvcs;
      fIntEr[i] =  sqrt( errF * errF + e_dvcs * e_dvcs );
      // -- Interference in A/B space --
      fReduced[i] = - ( QQ[iset] * TMath::Abs(t[iset]) ) / ( 2. * PI * Gamma * BuuI * GeV2nb) * ( F - bh - dvcs );
      fReducedEr[i] = TMath::Abs(- ( QQ[iset] * TMath::Abs(t[iset]) ) / ( 2. * PI * Gamma * BuuI * GeV2nb) * ( sqrt( errF * errF + e_dvcs * e_dvcs ) ) ); // Note: interf. error remain symmetric in A/B space

      if ( fAB[i] < AB_left_limit ) AB_left_limit = fAB[i];
      if ( fAB[i] > AB_right_limit ) AB_right_limit = fAB[i];

      //cout<<"["<<i<<"]: "<<"phi = "<<phi[i]<<", A/B = "<<fAB[i]<<", red_xs = "<<fReduced[i]<<endl;
    }

    // Ratio between A/B and C/B (A/C)
    fAC = new TF1("fAC", AuuICuuI_ratio, 0, 360, 4);
    fAC ->SetParameters(k, QQ[iset], xB[iset], t[iset]);
    fAC ->SetLineColor(1);
    fAC ->SetLineStyle(1);
    fAC ->SetLineWidth(2);

    //if ( TMath::Abs(fAC->Eval(180)) < 15 || TMath::Abs(fAC->Eval(0)) < 15) continue;

    //
    // cout<<"left: "<<AB_left_limit<<", right: "<<AB_right_limit<<endl;
    // -------------------------------------------------------------------------------------------

    // ----------------------- Rebining ----------------------------------------------------------
    // // Rebining in A/B space
    // // Method 1:
    // Step 1 -> Get the weighted average of the 2 points with the same A/B values
    for (Int_t i = 0; i < NumOfPts/2; i++) {

      fReduced_w[i] = ( 1. / ( fReducedEr[i] * fReducedEr[i] ) * fReduced[i] + 1. / ( fReducedEr[NumOfPts-1-i] * fReducedEr[NumOfPts-1-i] ) * fReduced[NumOfPts-1-i] ) /
                      ( 1. / ( fReducedEr[i] * fReducedEr[i] ) + 1. / ( fReducedEr[NumOfPts-1-i] * fReducedEr[NumOfPts-1-i] ) );
      fReducedEr_w[i] = 1. / sqrt( 1. / ( fReducedEr[i] * fReducedEr[i] ) + 1. / ( fReducedEr[NumOfPts-1-i] * fReducedEr[NumOfPts-1-i] ) );

      fAB_w[i] = fAB[i];
    }
    // fAB_w[(NumOfPts-1)/2] = fAB[NumOfPts-1];
    // fReduced_w[(NumOfPts-1)/2] = fReduced[NumOfPts-1];
    // fReducedEr_w[(NumOfPts-1)/2] = fReducedEr[NumOfPts-1];
    //
    // // -------------------------------------------------------------------------------------------
    //
    // // ----------------------- Fill graphs -------------------------------------------------------
    gABvsPhi[iset] = new TGraph(NumOfPts, fPhi, fAB);
    gABvsPhi[iset] ->SetName(Form("gABvsPhi_%d",iset));
    gABvsPhi[iset] ->SetTitle(Form("set %d: k = %.2f, QQ = %.2f, xB = %.2f, t = %.2f; #phi [deg];A_{UU}/B_{UU}", iset+1, k, QQ[iset], xB[iset], t[iset]));

    gIntf[iset] = new TGraphErrors(NumOfPts, fPhi, fInt, 0, fIntEr);
    gIntf[iset] ->SetName(Form("gIntf_%d",iset));
    gIntf[iset] ->SetTitle("BH-DVCS interference; #phi [deg];d^{4}#sigma [nb/GeV^{4}]");

    gMapped_Control[iset] = new TGraphErrors(NumOfPts, fAB, fReduced, 0, fReducedEr);
    gMapped_Control[iset] ->SetName(Form("gMapped_Control_%d",iset));
    gMapped_Control[iset] ->SetTitle(Form("set %d: k = %.2f, QQ = %.2f, xB = %.2f, t = %.2f; A^{I}_{UU}/B^{I}_{UU};d^{4}#sigma_{red}/B^{I}_{UU}", iset+1, k, QQ[iset], xB[iset], t[iset]));
    gMapped_Control[iset] ->Sort(); // Sort the graph points from low to high A/B values

    gMapped_Control_w[iset] = new TGraphErrors(NumOfPts/2, fAB_w, fReduced_w, 0, fReducedEr_w);
    gMapped_Control_w[iset] ->SetName(Form("gMapped_Control_w_%d",iset));
    gMapped_Control_w[iset] ->SetTitle(Form("set %d: k = %.2f, QQ = %.2f, xB = %.2f, t = %.2f; A^{I}_{UU}/B^{I}_{UU};d^{4}#sigma_{red}/B^{I}_{UU}", iset+1, k, QQ[iset], xB[iset], t[iset]));
    gMapped_Control_w[iset] ->Sort(); // Sort the graph points from low to high A/B values
    // -------------------------------------------------------------------------------------------

    // ----------------------- Fits -------------------------------------------------------
    fControlFit = new TF1("fControlFit", Linear_func, AB_left_limit, AB_right_limit, 3);
    fControlFit ->SetParNames("t", "ReH", "ReE");
    fControlFit ->SetLineColor(4);
    fControlFit ->SetLineStyle(1);
    fControlFit ->SetLineWidth(2);
    fControlFit ->FixParameter(0, t[iset]);

    gMapped_Control[iset] ->Fit(fControlFit,"R");

    fControlFit_w = new TF1("fControlFit_w", Linear_func, AB_left_limit, AB_right_limit, 3);
    fControlFit_w ->SetParNames("t", "ReH_w", "ReE_w");
    fControlFit_w ->SetLineColor(4);
    fControlFit_w ->SetLineStyle(1);
    fControlFit_w ->SetLineWidth(2);
    fControlFit_w ->FixParameter(0, t[iset]);

    gMapped_Control_w[iset] ->Fit(fControlFit_w,"R");

    // Simultaneous fit -----
    // Set simultaneous fit parameters from the initial fit on the xs vs phi space
		Double_t *pars = fTotalUUXS ->GetParameters();

    TF1 * ftotal_simult = new TF1("ftotal_simult", TotalUUXS, 0, 360, 8);
          ftotal_simult ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "dvcs");
          ftotal_simult ->SetLineColor(1);

    TF1 * fline_simult = new TF1("fline_simult", Linear_func, AB_left_limit, AB_right_limit, 3);
    fline_simult ->SetParNames("t", "ReH", "ReE");

    ROOT::Fit::FitResult fitresult;
		fitresult = SimultFit(gF[iset], gMapped_Control_w[iset], ftotal_simult, fline_simult, pars);

    const double *fitpars = fitresult.GetParams();
		const double *fitparserror = fitresult.GetErrors();

    cout<<"Simultaneous Fit ReH: "<< *(fitpars+4) <<" +- "<<*(fitparserror+4) <<endl;
    cout<<"Simultaneous Fit ReE: "<< *(fitpars+5) <<" +- "<<*(fitparserror+5) <<endl;

    // ----------------------- CFFs results ------------------------------------------------------
    gReH[0] ->SetPoint( iset, iset+1+0.05, fTotalUUXS->GetParameter(4) );
    gReH[0] ->SetPointError( iset, 0, fTotalUUXS->GetParError(4) );
    gReH[1] ->SetPoint( iset, iset+1+0.1, fControlFit_w ->GetParameter(1) );
    gReH[1] ->SetPointError( iset, 0, fControlFit_w ->GetParError(1) );
    gReH[2] ->SetPoint( iset, iset+1+0.15, *(fitpars+4) );
    gReH[2] ->SetPointError( iset, 0, *(fitparserror+4) );

    gReE[0] ->SetPoint( iset, iset+1+0.05, fTotalUUXS->GetParameter(5) );
    gReE[0] ->SetPointError( iset, 0, fTotalUUXS->GetParError(5) );
    gReE[1] ->SetPoint( iset, iset+1+0.1, fControlFit_w ->GetParameter(2) );
    gReE[1] ->SetPointError( iset, 0, fControlFit_w ->GetParError(2) );
    gReE[2] ->SetPoint( iset, iset+1+0.15, *(fitpars+5) );
    gReE[2] ->SetPointError( iset, 0, *(fitparserror+5) );

    gReHt[0] ->SetPoint( iset, iset+1+0.05, fTotalUUXS->GetParameter(6) );
    gReHt[0] ->SetPointError( iset, 0, fTotalUUXS->GetParError(6) );
    gReHt[2] ->SetPoint( iset, iset+1+0.15, *(fitpars+6) );
    gReHt[2] ->SetPointError( iset, 0, *(fitparserror+6) );

    // For Kin3 only (t for kin3)
    if ( 5 <= iset && iset <= 9 ){

        gReH_kin3[0] ->SetPoint( nkin3, -t[iset]+0.004, fTotalUUXS->GetParameter(4) );
        gReH_kin3[0] ->SetPointError(nkin3, 0, fTotalUUXS->GetParError(4) );
        gReH_kin3[1] ->SetPoint( nkin3, -t[iset]+0.008, fControlFit_w->GetParameter(1) );
        gReH_kin3[1] ->SetPointError(nkin3, 0, fControlFit_w->GetParError(1) );
        gReH_kin3[2] ->SetPoint( nkin3, -t[iset]+0.012, *(fitpars+4) );
        gReH_kin3[2] ->SetPointError( nkin3, 0, *(fitparserror+4) );

        gReE_kin3[0] ->SetPoint( nkin3, -t[iset], fTotalUUXS->GetParameter(5) );
        gReE_kin3[0] ->SetPointError( nkin3, 0, fTotalUUXS->GetParError(5) );
        gReE_kin3[1] ->SetPoint( nkin3, -t[iset]+0.004, fControlFit_w->GetParameter(2) );
        gReE_kin3[1] ->SetPointError(nkin3, 0, fControlFit_w->GetParError(2) );
        gReE_kin3[2] ->SetPoint( nkin3, -t[iset]+0.008, *(fitpars+5) );
        gReE_kin3[2] ->SetPointError( nkin3, 0, *(fitparserror+5) );

        gReHt_kin3[0] ->SetPoint( nkin3, -t[iset], fTotalUUXS->GetParameter(6) );
        gReHt_kin3[0] ->SetPointError(nkin3, 0, fTotalUUXS->GetParError(6) );
        gReHt_kin3[2] ->SetPoint( nkin3, -t[iset]+0.004, *(fitpars+6) );
        gReHt_kin3[2] ->SetPointError( nkin3, 0, *(fitparserror+6) );

        nkin3++;
    }



    // -------------------------------------------------------------------------------------------

    // ----------------------- Drawing the graphs ------------------------------------------------
    // Styles
    gF[iset] ->SetMarkerStyle(20);
    gIntf[iset] ->SetMarkerStyle(22);
    gIntf[iset] ->SetMarkerColor(2);
    gIntf[iset] ->SetLineColor(2);
    gABvsPhi[iset] ->SetMarkerStyle(20);
    gABvsPhi[iset] ->SetMarkerColor(2);
    gMapped_Control[iset] ->SetMarkerStyle(20);
    gMapped_Control_w[iset] ->SetMarkerStyle(20);

    TMultiGraph *mgr1 = new TMultiGraph();
    mgr1 ->SetTitle(Form("set %d: k = %.2f, QQ = %.2f, xB = %.2f, t = %.2f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", iset+1, k, QQ[iset], xB[iset], t[iset]));
    mgr1 ->Add(gF[iset], "P");
    mgr1 ->Add(gIntf[iset], "P");


    TText *tx = new TText(0.5,0.6,"");

    TCanvas *c1 = new TCanvas(Form("c1_%d",iset),"Rebining in A/B space", 4050,202,1881,1046);
    c1->Draw();
    auto *p1 = new TPad("p1","p1",0.,0.5,1.,1.);
    p1->Draw();
    p1->SetBottomMargin(0.001);

    p1->Divide(3,1);
    p1->cd(1);
    mgr1 ->Draw("ap");
    gPad->BuildLegend(0.19, 0.75, 0.4, 0.9);
    p1->cd(2);
    gABvsPhi[iset] ->Draw("ap");
    p1->cd(3);
    fAC ->Draw();

    c1->cd(0);
    auto *p2 = new TPad("p2","p3",0.,0.,1.,0.5);
    p2->Draw();
    p2->SetTopMargin(0.001);
    p2->SetBottomMargin(0.3);

    p2->Divide(3,1);
    p2->cd(1);
    gMapped_Control[iset] ->Draw("ap");

    TLegend* pLegend1 = new TLegend(0.58,0.77,0.9,0.9);
  				   pLegend1 ->SetTextSize(0.032);
  					 pLegend1 ->SetFillColor(10);
  				 	 pLegend1 ->SetBorderSize(1);
             pLegend1->SetHeader("    Fit values");
  					 pLegend1->AddEntry(tx,Form("ReH = %.2f #pm %.2f", fControlFit->GetParameter(1), fControlFit->GetParError(1)),"");
   					 pLegend1->AddEntry(tx,Form("ReE = %.2f #pm %.2f", fControlFit->GetParameter(2), fControlFit->GetParError(2)),"");

    pLegend1->Draw();

    p2->cd(2);
    gMapped_Control_w[iset] ->Draw("ap");

    TLegend* pLegend3 = new TLegend(0.58,0.77,0.9,0.9);
  				   pLegend3 ->SetTextSize(0.032);
  					 pLegend3 ->SetFillColor(10);
  				 	 pLegend3 ->SetBorderSize(1);
             pLegend3->SetHeader("    Fit values");
  					 pLegend3->AddEntry(tx,Form("ReH = %.2f #pm %.2f", fControlFit_w->GetParameter(1), fControlFit_w->GetParError(1)),"");
   					 pLegend3->AddEntry(tx,Form("ReE = %.2f #pm %.2f", fControlFit_w->GetParameter(2), fControlFit_w->GetParError(2)),"");

    pLegend3->Draw();



    if(iset == fNSet) break;
  } // end kinematics loop

  //  ----------------------------------- Draw CFFs graphs ------------------------------------------------
  // gReH_kin3rue ->SetLineColor(2);
  // gReH_kin3rue ->SetMarkerSize(0);
  TLine *line1 = new TLine(0.15,0,0.41,0);
  line1 ->SetLineColor(1);
  line1 ->SetLineStyle(3);
  TMultiGraph *mgr2 = new TMultiGraph();
  mgr2 ->SetTitle("ReH; set;ReH");
  mgr2 ->Add(gReH[0]);
  mgr2 ->Add(gReH[1]);
  mgr2 ->Add(gReH[2]);
  TMultiGraph *mgr3 = new TMultiGraph();
  mgr3 ->SetTitle("ReE; set;ReE");
  mgr3 ->Add(gReE[0]);
  mgr3 ->Add(gReE[1]);
  mgr3 ->Add(gReE[2]);
  TMultiGraph *mgr4 = new TMultiGraph();
  mgr4 ->SetTitle("ReHtilde; set;ReHtilde");
  mgr4 ->Add(gReHt[0]);
  mgr4 ->Add(gReHt[2]);
  TMultiGraph *mgr5 = new TMultiGraph();
  mgr5 ->SetTitle("; -t [GeV^{2}]; ReH");
  mgr5 ->Add(gReH_kin3[0]);
  // mgr5 ->Add(gReH_kin3[1]);
  mgr5 ->Add(gReH_kin3[2]);
  mgr5 ->Add(gReH_ref_KM15);
  TMultiGraph *mgr6 = new TMultiGraph();
  mgr6 ->SetTitle("; -t [GeV^{2}]; ReE");
  mgr6 ->Add(gReE_kin3[0]);
  // mgr6 ->Add(gReE_kin3[1]);
  mgr6 ->Add(gReE_kin3[2]);
  TMultiGraph *mgr7 = new TMultiGraph();
  mgr7 ->SetTitle("; -t [GeV^{2}]; ReHtilde");
  mgr7 ->Add(gReHt_kin3[0]);
  mgr7 ->Add(gReHt_kin3[2]);

  // without the local fit since its erros are very large
  TMultiGraph *mgr8 = new TMultiGraph();
  mgr8 ->SetTitle("; -t [GeV^{2}]; ReH");
  mgr8 ->Add(gReH_kin3[1]);
  mgr8 ->Add(gReH_kin3[2]);
  mgr8 ->Add(gReH_ref_KM15);
  TMultiGraph *mgr9 = new TMultiGraph();
  mgr9 ->SetTitle("; -t [GeV^{2}]; ReE");
  mgr9 ->Add(gReE_kin3[1]);
  mgr9 ->Add(gReE_kin3[2]);
  TMultiGraph *mgr10 = new TMultiGraph();
  mgr10 ->SetTitle("; -t [GeV^{2}]; ReHtilde");
  mgr10 ->Add(gReHt_kin3[2]);


  TCanvas * c2 = new TCanvas("c2","ReH", 4446,332,1685,899);
  mgr2 ->Draw("ap");
  gPad->BuildLegend(0.25, 0.78, 0.35, 0.9);

  TCanvas * c3 = new TCanvas("c3","ReE", 4446,332,1685,899);
  mgr3 ->Draw("ap");
  gPad->BuildLegend(0.25, 0.78, 0.35, 0.9);

  TCanvas * c4 = new TCanvas("c4","ReHtilde", 4446,332,1685,899);
  mgr4 ->Draw("ap");
  gPad->BuildLegend(0.25, 0.78, 0.35, 0.9);

  TLegend* pLegend1 = new TLegend(0.23, 0.75, 0.48, 0.9);
				   pLegend1 ->SetTextSize(0.032);
					 pLegend1 ->SetFillColor(10);
				 	 pLegend1 ->SetBorderSize(1);
					 pLegend1->AddEntry(gReH_kin3[0],"Local fit","LP");
 					 // pLegend1->AddEntry(gReH_kin3[1],"Linear fit","LP");
 				   pLegend1->AddEntry(gReH_kin3[2],"Simultaneous","LP");
					 pLegend1->AddEntry(gReH_ref_KM15,"KM15","LP");

  TLegend* pLegend2 = new TLegend(0.53, 0.75, 0.78, 0.9);
 				   pLegend2 ->SetTextSize(0.032);
 					 pLegend2 ->SetFillColor(10);
 				 	 pLegend2 ->SetBorderSize(1);
					 pLegend2->AddEntry(gReH_kin3[0],"Local fit","LP");
				   pLegend2->AddEntry(gReH_kin3[2],"Simultaneous","LP");
 					 // pLegend2->AddEntry(gReH_ref_KM15,"KM15","LP");

  TCanvas *c5 = new TCanvas("c5","ReH_kin3",4586,502,1071,741);
  mgr5 ->Draw("AP");
  // // mgr5 ->GetHistogram()->SetMinimum(-7.9);
  // // mgr5 ->GetHistogram()->SetMaximum(9.9);
  mgr5->GetXaxis()->SetLimits(0.141, 0.41);
  //gPad->BuildLegend(0.25, 0.78, 0.35, 0.9);
  pLegend1 ->Draw();
  //line1->Draw("same");
  gPad->Update();

  TCanvas * c6 = new TCanvas("c6","ReE_kin3",4586,502,1071,741);
  mgr6 ->Draw("ap");
  //line1->Draw("same");
  mgr6->GetXaxis()->SetLimits(0.141, 0.41);
  gPad->BuildLegend(0.2, 0.76, 0.35, 0.9);

  TCanvas * c7 = new TCanvas("c7","ReHt_kin3",4586,502,1071,741);
  mgr7 ->Draw("ap");
  //line1->Draw("same");
  mgr7->GetXaxis()->SetLimits(0.141, 0.41);
  gPad->BuildLegend(0.25, 0.76, 0.4, 0.9);

  TCanvas * ckin3 = new TCanvas("ckin3","CFFs", 4446,332,1685,899);
  ckin3->Divide(3,2);
  ckin3->cd(1);
  mgr5 ->Draw("ap");
  line1->Draw("same");
  mgr5->GetXaxis()->SetLimits(0.141, 0.41);
  pLegend1 ->Draw();
  ckin3->cd(2);
  mgr6 ->Draw("ap");
  line1->Draw("same");
  line1->Draw("same");
  mgr6->GetXaxis()->SetLimits(0.141, 0.41);
  gPad->BuildLegend(0.15, 0.76, 0.35, 0.9);
  ckin3->cd(3);
  mgr7 ->Draw("ap");
  line1->Draw("same");
  mgr7->GetXaxis()->SetLimits(0.141, 0.41);
  gPad->BuildLegend(0.25, 0.76, 0.45, 0.9);
  ckin3->cd(4);
  mgr8 ->Draw("ap");
  line1->Draw("same");
  mgr8->GetXaxis()->SetLimits(0.141, 0.41);
  mgr8 ->GetHistogram()->SetMaximum(1.8);
  mgr8 ->GetHistogram()->SetMinimum(-6.5);
  pLegend2 ->Draw();
  ckin3->cd(5);
  mgr9 ->Draw("ap");
  line1->Draw("same");
  mgr9->GetXaxis()->SetLimits(0.141, 0.41);
  mgr9 ->GetHistogram()->SetMaximum(18);
  mgr9 ->GetHistogram()->SetMinimum(-22);
  gPad->BuildLegend(0.18, 0.76, 0.4, 0.9);
  ckin3->cd(6);
  mgr10 ->Draw("ap");
  line1->Draw("same");
  mgr10->GetXaxis()->SetLimits(0.141, 0.41);
  mgr10 ->GetHistogram()->SetMaximum(14);
  mgr10 ->GetHistogram()->SetMinimum(-14);
  gPad->BuildLegend(0.2, 0.76, 0.48, 0.9);

  ckin3 ->Print("CFFs_HallA_Kin3.pdf)","pdf");

  TCanvas * ckin3_2 = new TCanvas("ckin3_2","CFFs",4381,369,1750,556);

  ckin3_2->Divide(3,1);

  ckin3_2->cd(1);
  gPad->SetBorderSize(1);
	gPad->SetLeftMargin(0.12);
	gPad->SetRightMargin(0.03);
  mgr5 ->Draw("ap");
  line1->Draw("same");
  mgr5->GetXaxis()->SetLimits(0.141, 0.41);
  pLegend1 ->Draw();
  ckin3_2->cd(2);
  gPad->SetBorderSize(1);
	gPad->SetLeftMargin(0.12);
	gPad->SetRightMargin(0.03);
  mgr6 ->Draw("ap");
  line1->Draw("same");
  mgr6->GetXaxis()->SetLimits(0.141, 0.41);
  pLegend2 ->Draw();
  ckin3_2->cd(3);
  gPad->SetBorderSize(1);
	gPad->SetLeftMargin(0.12);
	gPad->SetRightMargin(0.03);
  mgr7 ->Draw("ap");
  line1->Draw("same");
  mgr7->GetXaxis()->SetLimits(0.141, 0.41);
  pLegend2 ->Draw();

  ckin3_2 ->Print("CFFs_HallA_Kin3_local_2.pdf)","pdf");

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
//_________________________________________________________________________
Double_t Linear_func(Double_t *argx, Double_t *par)
{
	const Double_t x = argx[0];
	Double_t _t = par[0];
	Double_t ReH = par[1];
	Double_t ReE = par[2];

	Double_t _tau = -0.25 * _t / (M*M);
	Double_t _F1 = ff->ffF1_K(_t);
	Double_t _F2 = ff->ffF2_K(_t);

	Double_t m = _F1 * ReH + _tau * _F2 * ReE;
	Double_t n = ( _F1 + _F2 ) * ( ReH + ReE );

	Double_t f = m * x + n;

	return f;
}
//_________________________________________________________________________
Double_t AuuICuuI_ratio(Double_t *angle, Double_t *par)
{
  Double_t _phi = angle[0];

  Double_t _k = par[0];
  Double_t _QQ = par[1];
  Double_t _xB = par[2];
  Double_t _t = par[3];

	// Set QQ, xB, t and k
	tva1 ->SetKinematics( _QQ, _xB, _t, _k );

  Double_t AuuI, BuuI, CuuI;

  // Check ratio of AuuI/BuuI and CuuI/BuuI
  tva1->GetIUUCoefficients(_phi, AuuI, BuuI, CuuI);

	return AuuI/CuuI;
}
//_________________________________________________________________________
ROOT::Fit::FitResult SimultFit(TGraphErrors* g1, TGraphErrors* g2, TF1 *f1, TF1 *f2, Double_t *pars)
{
	// perform global fit
	ROOT::Math::WrappedMultiTF1 wfB1(*f1,1);
	ROOT::Math::WrappedMultiTF1 wfB2(*f2,1);

	ROOT::Fit::DataOptions opt;
  ROOT::Fit::DataRange rangeB1;
  ROOT::Fit::DataRange rangeB2;

  // // set the data range
  // rangeB1.SetRange(0,360);
  // rangeB2.SetRange(0,200);

	ROOT::Fit::BinData dataB1(opt,rangeB1);
	ROOT::Fit::BinData dataB2(opt,rangeB2);

	ROOT::Fit::FillData(dataB1, g1);
	ROOT::Fit::FillData(dataB2, g2);

	ROOT::Fit::Chi2Function chi2_B1(dataB1, wfB1);
  ROOT::Fit::Chi2Function chi2_B2(dataB2, wfB2);

  GlobalChi2 globalChi2(chi2_B1, chi2_B2);

  ROOT::Fit::Fitter fitter;

	const int Npar = 8;
  //double par0[Npar] = { 2.193, 0.342, -0.3710, 5.75, 1, 0, 1, 0.00592806};
	// Initialize parameters
	Double_t par0[Npar];
	for (Int_t i = 0; i < Npar; i++) par0[i] = *(pars+i);

  // create before the parameter settings in order to fix or set range on them
  fitter.Config().SetParamsSettings(8,par0);

	//cout<< "Initial Parameters: "<<par0[0]<<", "<<par0[1]<<", "<<par0[2]<<", "<<par0[3]<<", "<<par0[4]<<", "<<par0[5]<<", "<<par0[6]<<endl;

	// fix parameters
  fitter.Config().ParSettings(0).Fix();
	fitter.Config().ParSettings(1).Fix();
	fitter.Config().ParSettings(2).Fix();
	fitter.Config().ParSettings(3).Fix();
  // set limits on the third and 4-th parameter
  //fitter.Config().ParSettings(2).SetLimits(-10,-1.E-4);

	fitter.Config().MinimizerOptions().SetPrintLevel(0);
  fitter.Config().SetMinimizer("Minuit","Minimize");

  // fit FCN function directly
  // (specify optionally data size and flag to indicate that is a chi2 fit)
  fitter.FitFCN(8,globalChi2,0,dataB1.Size()+dataB2.Size(),true);
  ROOT::Fit::FitResult result = fitter.Result();
  result.Print(std::cout);

	return result;
}
