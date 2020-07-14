#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include "TBHDVCS.h"
#include "TBHDVCS.cxx"
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
#include "TLine.h"
#include "TStyle.h"

using namespace std;

TBHDVCS *bhdvcs = new TBHDVCS;

TFormFactors *ff = new TFormFactors;

//____________________________________________________________________________________________________
Double_t TotalUUXS(Double_t *angle, Double_t *par)	{// Total Cross Section - user defined function

	Double_t _phi = angle[0];
	Double_t _k = par[0];
	Double_t _QQ = par[1];
	Double_t _xB = par[2];
	Double_t _t = par[3];
	Double_t _ReH = par[4];
	Double_t _ReE = par[5];
	Double_t _ReHtilde = par[6];
	Double_t _dvcs = par[7];

	Double_t _F1 = ff->ffF1(_t);
	Double_t _F2 = ff->ffF2(_t);
	Double_t AUUI, BUUI, CUUI;

	// Set QQ, xB, t and k
	bhdvcs ->SetKinematics( _QQ, _xB, _t, _k );

	Double_t xsbhuu	 = bhdvcs ->GetBHUUxs(_phi, _F1, _F2); // BH cross section
	Double_t xsiuu = bhdvcs ->GetIUUxs(_phi, _F1, _F2, _ReH, _ReE, _ReHtilde, AUUI, BUUI, CUUI);

	Double_t tot_sigma_uu = xsbhuu + xsiuu + _dvcs; // Constant added to account for DVCS contribution

	return tot_sigma_uu;
}
//___________________________________________________________________________________________________
TGraph* shade(TF1 *f1, TF1 *f2, Double_t xmin, Double_t xmax) { //shade the area between f1 and f2

	//create a TGraph to store the function values
	TGraph *gr = new TGraph();
	gr ->SetFillColor(kYellow);
	//gr ->SetFillStyle(3001);
	//gr ->SetFillStyle(3002);

	//process first function
	Int_t npx = f1->GetNpx();
	Int_t npoints=0;
	Double_t dx = (xmax-xmin)/npx;
	Double_t x = xmin+0.5* dx;
	while (x <= xmax) {
		Double_t y = f1->Eval(x);
		//if (y < ymin) y = ymin;
		//if (y > ymax) y = ymax;
		gr->SetPoint(npoints,x,y);
		npoints++;
		x += dx;
	}
	//process second function
	x = xmax-0.5 *dx;
	while (x >= xmin) {
		Double_t y = f2->Eval(x);
		//if (y < ymin) y = ymin;
		//if (y > ymax) y = ymax;
		gr->SetPoint(npoints,x,y);
		npoints++;
		x -= dx;
	}
	return gr;
}
// Draw Graphs
//___________________________________________________________________________________________________
TCanvas *DrawGraphs(Long64_t nsets, TGraph* gGen, TGraphErrors* gRec, TString WhichCFFs, Bool_t ifSelect) {

	gStyle->SetOptStat(0);

	TCanvas* c;

	if (ifSelect)
	c = new TCanvas(Form("c%s_sel",WhichCFFs.Data() ),"3 Graphs",552,348,1622,1598);
	else
	c = new TCanvas(Form("c%s",WhichCFFs.Data() ),"3 Graphs",552,348,1622,1598);

	auto *p2 = new TPad("p2","p3",0.,0.,1.,0.3); p2->Draw();
	p2->SetTopMargin(0.001);
	p2->SetBottomMargin(0.3);
	//p2->SetLogx ();
	//p2->SetGrid();

	auto *p1 = new TPad("p1","p1",0.,0.3,1.,1.);  p1->Draw();
	p1->SetBottomMargin(0.001);
	p1->cd();
	//p1->SetGrid();
	//p1->SetLogx();

	gGen->SetMarkerColor(kRed);
	gGen->SetMarkerStyle(20);
	gGen->SetMarkerSize(1.8);
	gGen->SetTitle("");

	gGen->SetTitle("");
	gGen->GetXaxis()->SetTitle("kin");
	gGen->GetXaxis()->SetLimits(0.01, nsets+1.);
	//gGen->GetYaxis()->CenterTitle();
	gGen->GetYaxis()->SetTitleOffset(1.5);
	gGen->GetXaxis()->SetTickSize(0.);

	if (WhichCFFs == "ReH") {
		gGen->GetYaxis()->SetTitle("ReH");
		gGen->GetYaxis()->SetRangeUser(-19,37);	}
	if (WhichCFFs == "ReE") {
		gGen->GetYaxis()->SetTitle("ReE");
		gGen->GetYaxis()->SetRangeUser(-125,39);	}
	if (WhichCFFs == "ReHtilde") {
		gGen->GetYaxis()->SetTitle("ReHtilde");
		gGen->GetYaxis()->SetRangeUser(-9,19);	}

	gGen->Draw("AP");

	gRec->SetMarkerColor(kBlack);
	gRec->SetMarkerStyle(24);
	gRec->SetMarkerSize(1.8);
	gRec->Draw("P");

	TLegend *leg = new TLegend(0.15,0.78,0.32,0.87);

	leg->AddEntry(gGen,"Generated","p");
	leg->AddEntry(gRec,"Reconstructed","lp");
	leg->Draw();

	// ratio
	p2->cd();

	// 20 % area band
	TF1* f1 = new TF1("f1", "[0]*x + [1]", 0.5, nsets+1.);
	TF1* f2 = new TF1("f2", "[0]*x + [1]", 0.5, nsets+1.);
	f1 ->SetParameters(0,20);
	f2 ->SetParameters(0,-20);
	TGraph* gshade = new TGraph();
	gshade = shade(f1, f2, 0.5, nsets+1.);

	//TLine* ln = new TLine(0, 1, 21, 1);
	TH1F* hEmpty;
	if(ifSelect)
	hEmpty  = new TH1F(Form("h%s_sel",WhichCFFs.Data()),"", 20, 0.01, nsets+1.);
	else
	hEmpty  = new TH1F(Form("h%s",WhichCFFs.Data()),"", 20, 0.01, nsets+1.);
	hEmpty ->SetTitle("");
	hEmpty ->SetMinimum(-198);
	hEmpty ->SetMaximum(198);
	hEmpty->GetYaxis()->SetLabelSize(0.075);
	hEmpty->GetYaxis()->SetLabelSize(0.075);
	hEmpty->GetXaxis()->SetLabelSize(0.075);
	hEmpty->GetXaxis()->SetLabelSize(0.075);
	hEmpty->GetYaxis()->SetTitleSize(0.06);
	hEmpty->GetYaxis()->SetTitleOffset(0.8);
	hEmpty->SetYTitle("#frac{Rec - True}{#lbar True #lbar} %	");
	//hEmpty->SetYTitle("ratio");
	hEmpty->GetXaxis()->SetTitleSize(.1);
	hEmpty->SetXTitle("set");

	Int_t n20 = 0;

	TGraph*r = new TGraph(nsets); r->SetTitle("");
	for (int i=0; i<nsets; i++){
		Double_t pct_change = 100. * (gRec ->GetPointY(i) - gGen ->GetPointY(i)) / TMath::Abs( gGen ->GetPointY(i) );
		r->SetPoint( i, gGen ->GetPointX(i), pct_change ); // Percent Change
		if ( TMath::Abs(pct_change) <= 20. ) n20++;
		//r->SetPoint( i, gGen ->GetPointX(i), gRec ->GetPointY(i)/gGen ->GetPointY(i) ); // ratio
	}

	r->SetMarkerStyle(20);
	r->SetMarkerSize(1.8);
	hEmpty ->Draw("0");
	//ln ->Draw("same");
	gshade ->Draw("fsame0");
	r->Draw("ZPsame");

	TText *tx = new TText(0.5,0.6,"");
	TLegend *leg2 = new TLegend(0.75,0.88,0.89,0.96);
	leg2->SetTextSize(0.06);
	leg2->SetHeader(Form(" %d sets within 20%% ",n20));
	leg2->Draw();

	return c;

}
//___________________________________________________________________________________________________
void DoROOTFit(TGraphErrors* gGenData, TF1* ffit ) { //Fit with ROOT::Fit method

	ROOT::Fit::DataOptions opt;
	ROOT::Fit::DataRange range;
	// set the data range
	range.SetRange(0,360);

	ROOT::Fit::BinData data(opt,range);
	ROOT::Fit::FillData(data, gGenData);

	ROOT::Math::WrappedMultiTF1 fitfunc(*ffit,1);
	ROOT::Fit::Fitter fitter;
	fitter.SetFunction(fitfunc, false);

	// fix parameters
	fitter.Config().ParSettings(0).Fix();
	fitter.Config().ParSettings(1).Fix();
	fitter.Config().ParSettings(2).Fix();
	fitter.Config().ParSettings(3).Fix();
	fitter.Config().ParSettings(7).Fix();

	ROOT::Math::MinimizerOptions(mopt);
	mopt.SetMinimizerType("Minuit2");
	mopt.SetMinimizerAlgorithm("Minimize");
	// print the default minimizer option values
	mopt.Print();

	fitter.Fit(data); // chi2 fit

	ROOT::Fit::FitResult result = fitter.Result();
	result.Print(std::cout);
	ffit ->SetFitResult(result);
}
//___________________________________________________________________________________________________
void AnaBHDVCS(TString DataFileName, Bool_t bDrawFits) {

	Int_t nPoints = 36; // Size of the generated data chunks
	Double_t var = 0.05; // Variance of F

	TFile* myFile = TFile::Open(DataFileName);

	// Create a TTreeReader for the tree, by passing the TTree's name and the  TFile it is in.
	TTreeReader myReader("dvcs", myFile);

	TTreeReaderValue<Double_t> myk(myReader, "kinematics.k");
	TTreeReaderValue<Double_t> myQQ(myReader, "kinematics.QQ");
	TTreeReaderValue<Double_t> myxB(myReader, "kinematics.xB");
	TTreeReaderValue<Double_t> myt(myReader, "kinematics.t");
	TTreeReaderValue<Double_t> mydvcs(myReader, "dvcs");
	TTreeReaderArray<Double_t> phi(myReader, "phi");
	TTreeReaderArray<Double_t> F(myReader, "F");
	TTreeReaderArray<Double_t> errF(myReader, "errF");
	TTreeReaderValue<Double_t> myvarF(myReader, "varF");
	TTreeReaderValue<Double_t> genReH(myReader, "ReH");
	TTreeReaderValue<Double_t> genReE(myReader, "ReE");
	TTreeReaderValue<Double_t> genReHtilde(myReader, "ReHtilde");

	Int_t iset = 0;
	Int_t nsel = 0;

	Long64_t NumOfSets = myReader.GetEntries();
	cout<<"Number of sets: "<<NumOfSets<<endl;

	// Fit Function initialization
	TF1* fTotalUUXS[NumOfSets];

	// Graphs of the total xs vs phi
	TGraphErrors* gGenDVCS[NumOfSets];
	// CFFs graphs
	TGraph* gGenReH = new TGraph(NumOfSets);
	TGraph* gGenReE = new TGraph(NumOfSets);
	TGraph* gGenReHtilde = new TGraph(NumOfSets);
	TGraphErrors* gRecReH = new TGraphErrors(NumOfSets);
	TGraphErrors* gRecReE = new TGraphErrors(NumOfSets);
	TGraphErrors* gRecReHtilde = new TGraphErrors(NumOfSets);

	// Filter kinematic settings
	TGraph* gGenReH_sel = new TGraph();
	TGraph* gGenReE_sel = new TGraph();
	TGraph* gGenReHtilde_sel = new TGraph();
	TGraphErrors* gRecReH_sel = new TGraphErrors();
	TGraphErrors* gRecReE_sel = new TGraphErrors();
	TGraphErrors* gRecReHtilde_sel = new TGraphErrors();

	ofstream myfile; //output file
	myfile.open ("dvcs_xs_newsets_withCFFs.csv");
	myfile<<"#Set,index,k,QQ,x_b,t,phi_x,F,errF,F1,F2,dvcs,ReH_true,ReE_true,ReHtilde_true,ReH_fit,ReE_fit,ReHtilde_fit"<<endl;

	// Loop through all the TTree's entries
	while (myReader.Next()) {
	  // behaves like an iterator
		Double_t k = *myk;
		Double_t QQ = *myQQ;
		Double_t xB = *myxB;
		Double_t t = *myt;
		Double_t dvcs = *mydvcs;
		Double_t varF = *myvarF;
		Double_t ReH = *genReH;
		Double_t ReE = *genReE;
		Double_t ReHtilde = *genReHtilde;

		fTotalUUXS[iset] = new TF1(Form("fTotalUUXS_%d",iset), TotalUUXS, 0, 360, 8);
		fTotalUUXS[iset] ->SetParNames("k", "QQ", "xB", "t", "ReH", "ReE", "ReHtilde", "DVCSXS");
		fTotalUUXS[iset] ->SetLineColor(4);
		fTotalUUXS[iset] ->SetLineStyle(1);
		fTotalUUXS[iset] ->SetLineWidth(2);

		// Set Kinematics on the fit function
		fTotalUUXS[iset] ->FixParameter(0, k);
		fTotalUUXS[iset] ->FixParameter(1, QQ);
		fTotalUUXS[iset] ->FixParameter(2, xB);
		fTotalUUXS[iset] ->FixParameter(3, t);
		fTotalUUXS[iset] ->FixParameter(7, dvcs);

		gGenDVCS[iset] = new TGraphErrors(36);
		gGenDVCS[iset] ->SetName(Form("gGenDVCS_%d",iset));
		gGenDVCS[iset] ->SetTitle(Form("k = %.2f, QQ = %.2f, xB = %.2f, t = %.2f; #phi [deg];d^{4}#sigma [nb/GeV^{4}]", k, QQ, xB, t));
		gGenDVCS[iset] ->SetMarkerStyle(20);
		gGenDVCS[iset] ->SetMarkerSize(1);

		for (Int_t i = 0; i < nPoints; i++) {
			// Fill pseudo data graph
			gGenDVCS[iset] ->SetPoint( i, phi[i], F[i] );
			gGenDVCS[iset] ->SetPointError( i, 0, errF[i] );
		}

		//Fit with ROOT::Fit
		DoROOTFit( gGenDVCS[iset], fTotalUUXS[iset] );

		Double_t recReH, recReE, recReHtilde;
		Double_t recdeltaReH, recdeltaReE, recdeltaReHtilde;

		recReH = fTotalUUXS[iset] ->GetParameter(4); //ReH
		recReE = fTotalUUXS[iset] ->GetParameter(5); //ReE
		recReHtilde = fTotalUUXS[iset] ->GetParameter(6); //ReHtilde
		recdeltaReH = fTotalUUXS[iset] ->GetParError(4); //ReH fit error
		recdeltaReE = fTotalUUXS[iset] ->GetParError(5); //ReE fit error
		recdeltaReHtilde = fTotalUUXS[iset] ->GetParError(6); //ReHtilde fit error

		gGenReH ->SetPoint(iset, iset+1, ReH);
		gGenReE ->SetPoint(iset, iset+1, ReE);
		gGenReHtilde ->SetPoint(iset, iset+1, ReHtilde);

		gRecReH ->SetPoint(iset, iset+1, recReH);
		gRecReH ->SetPointError(iset, 0, recdeltaReH);

		gRecReE ->SetPoint(iset, iset+1, recReE);
		gRecReE ->SetPointError(iset, 0, recdeltaReE);

		gRecReHtilde ->SetPoint(iset, iset+1, recReHtilde);
		gRecReHtilde ->SetPointError(iset, 0, recdeltaReHtilde);

		// Select good kinematics (< 20 %)
		Double_t pct_change_ReH = 100. * (recReH - ReH) / TMath::Abs( ReH );
		Double_t pct_change_ReE = 100. * (recReE - ReE) / TMath::Abs( ReE );
		Double_t pct_change_ReHtilde = 100. * (recReHtilde - ReHtilde) / TMath::Abs( ReHtilde );

		if ( (TMath::Abs(pct_change_ReH) <= 20.) && (TMath::Abs(pct_change_ReE) <= 20.) && (TMath::Abs(pct_change_ReHtilde) <= 20.) ) {

			gGenReH_sel->SetPoint( nsel, nsel+1, ReH );
			gRecReH_sel->SetPoint( nsel, nsel+1, recReH );
			gRecReH_sel->SetPointError( nsel, 0, recdeltaReH );

			gGenReE_sel->SetPoint( nsel, nsel+1, ReE );
			gRecReE_sel->SetPoint( nsel, nsel+1, recReE );
			gRecReE_sel->SetPointError( nsel, 0, recdeltaReE );

			gGenReHtilde_sel->SetPoint( nsel, nsel+1, ReHtilde );
			gRecReHtilde_sel->SetPoint( nsel, nsel+1, recReHtilde );
			gRecReHtilde_sel->SetPointError( nsel, 0, recdeltaReHtilde );

			// // Drawing the graph
			if (bDrawFits){
				TCanvas * c1 = new TCanvas(Form("c_%d",iset),"pseudo data fit", 552, 274, 2198, 1710);
				gStyle->SetOptFit(1111);
				fTotalUUXS[iset]->SetRange(0, 360);
				fTotalUUXS[iset]->SetLineColor(kBlue);
				gGenDVCS[iset] ->GetListOfFunctions()->Add(fTotalUUXS[iset]);
				gGenDVCS[iset] ->Draw("ap");
			}

			// print output File
			for (Int_t i = 0; i < nPoints; i++) {
				myfile<<nsel<<","<<i<<","<<k<<","<<QQ<<","<<xB<<","<<t<<","<<phi[i]<<","<<F[i]<<","<<errF[i]<<","<<ff->ffF1(t)<<","<<ff->ffF2(t)<<","<<dvcs<<","<<ReH<<","
							<<ReE<<","<<ReHtilde<<","<<recReH<<","<<recReE<<","<<recReHtilde<<endl;
			}
			nsel++;
		} // end if
		iset++;
	} // end tree entries loop

	// Draw ratio graphs
	TCanvas* c4 = DrawGraphs(nsel, gGenReH_sel, gRecReH_sel,"ReH",kTRUE);
					 c4->Print(Form("./CFFs_SelKine_%.2f.pdf(",var),"pdf");
	TCanvas* c5 = DrawGraphs(nsel, gGenReE_sel, gRecReE_sel,"ReE",kTRUE);
					 c5->Print(Form("./CFFs_SelKine_%.2f.pdf",var),"pdf");
	TCanvas* c6 = DrawGraphs(nsel, gGenReHtilde_sel, gRecReHtilde_sel,"ReHtilde",kTRUE);
					 c6->Print(Form("./CFFs_SelKine_%.2f.pdf)",var),"pdf");

}
