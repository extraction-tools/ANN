#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include "TROOT.h"
#include "TStyle.h"
#include "TChain.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TH1I.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TF1.h"
#include "TTree.h"
//#include "defineHistos.h"
#include "TH1F.h"
#include "TLorentzVector.h"



using namespace std; 


Double_t QQ;
Double_t xb;
Double_t t;
Double_t k;
Double_t F1;
Double_t F2;
Double_t dvcs;
Double_t ReH_fit[342];
Double_t ReH_err[342];
Double_t ReE_fit[342];
Double_t ReE_err[342];
Double_t ReHTilde_fit[342];
Double_t ReHTilde_err[342];


Double_t fitFunction2(Double_t *x, Double_t *par); 
Double_t fitFunction3(Double_t *x, Double_t *par); 
//{
double TProduct (TLorentzVector v1, TLorentzVector v2) {
	// Transverse product
  Double_t tv1v2;
  return tv1v2 = v1.Px() * v2.Px() + v1.Py() * v2.Py();
}


Double_t fitFunction(Double_t *x, Double_t *par) 
{

const double p0 = par[0];
const double p1 = par[1];
const double p2 = par[2];
const double phi = x[0];



Double_t ALP_INV = 137.0359998; 
Double_t PI = 3.1415926535;
Double_t RAD = PI / 180.;
Double_t M = 0.938272; 
Double_t GeV2nb = .389379*1000000; 

Double_t chisq = 0.0;
Double_t function = 0.0;
Double_t prob = 0.0;
Double_t nll = 0.0;


Double_t y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau;

  
TLorentzVector K, KP, Q, QP, D, p, P;
Double_t kkp, kq, kp, kpp;
Double_t kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd;
Double_t kk_T, kqp_T, kkp_T, kd_T, dd_T, kpqp_T, kP_T, kpP_T, qpP_T, kpd_T, qpd_T, Dplus, Dminus;

Double_t s;     
Double_t Gamma; 
Double_t jcob;  
Double_t AUUBH, BUUBH; 
Double_t AUUI, BUUI, CUUI; 
Double_t con_AUUBH, con_BUUBH, con_AUUI, con_BUUI, con_CUUI;  
Double_t bhAUU, bhBUU; 
Double_t iAUU, iBUU, iCUU; 
Double_t real_iAUU, real_iBUU, real_iCUU; 
Double_t xbhUU;
Double_t xIUU; 
Double_t real_xbhUU;
Double_t real_xIUU; 




M2 = M*M; 
y = QQ / ( 2. * M * k * xb ); 
  
gg = 4. * M2 * xb * xb / QQ; 
e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); 
xi = 1. * xb * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - xb + xb * t / QQ ) ); 
tmin = -( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( xb * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* xb ) ) ); 
kpr = k * ( 1. - y ); 

qp = t / 2. / M + k - kpr; 
po = M - t / 2. / M; 
pmag = sqrt( ( -t ) * ( 1. - t / 4. / M / M ) ); 
cth = -1. / sqrt( 1. + gg ) * ( 1. + gg / 2. * ( 1. + t / QQ ) / ( 1. + xb * t / QQ ) ); 
theta = acos(cth); 
sthl = sqrt( gg ) / sqrt( 1. + gg ) * ( sqrt ( 1. - y - y * y * gg / 4. ) ); 
cthl = -1. / sqrt( 1. + gg ) * ( 1. + y * gg / 2. ) ; 
tau = -0.25 * t / M2;
K.SetPxPyPzE( k * sthl, 0.0, k * cthl, k );
KP.SetPxPyPzE( K(0), 0.0, k * ( cthl + y * sqrt( 1. + gg ) ), kpr );
Q = K - KP;
p.SetPxPyPzE(0.0, 0.0, 0.0, M);

s = (p + K) * (p + K);
Gamma = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / ( s - M2 ) / ( s - M2 ) / sqrt( 1. + gg ) / xb;
jcob = 2. * PI ;

//
QP.SetPxPyPzE(qp * sin(theta) * cos( RAD*phi ), qp * sin(theta) * sin( RAD*phi ), qp * cos(theta), qp);
D = Q - QP; 
TLorentzVector pp = p + D; 
P = p + pp;
P.SetPxPyPzE(.5*P.Px(), .5*P.Py(), .5*P.Pz(), .5*P.E());

//
kkp  = K * KP;   
kq   = K * Q;    
kp   = K * p;    
kpp  = KP * p;   

kd   = K * D;  
kpd  = KP * D;   
kP   = K * P;    
kpP  = KP * P;   
kqp  = K * QP;   
kpqp = KP * QP;  
dd   = D * D;    
Pq   = P * Q;    
Pqp  = P * QP;   
qd   = Q * D;    
qpd  = QP * D;   

//

kk_T   = TProduct(K,K);
  kkp_T  = kk_T;
  kqp_T  = TProduct(K,QP);
  kd_T   = -1.* kqp_T;
  dd_T   = TProduct(D,D);
  kpqp_T = kqp_T;
  kP_T   = TProduct(K,P);
  kpP_T  = TProduct(KP,P);
  qpP_T  = TProduct(QP,P);
  kpd_T  = -1.* kqp_T;
  qpd_T  = -1. * dd_T;

  Dplus   = .5 / kpqp - .5 / kqp;
  Dminus  = -.5 / kpqp - .5 / kqp;


//
AUUBH = ( (8. * M2) / (t * kqp * kpqp) ) * ( (4. * tau * (kP * kP + kpP * kpP) ) - ( (tau + 1.) * (kd * kd + kpd * kpd) ) );
BUUBH = ( (16. * M2) / (t* kqp * kpqp) ) * (kd * kd + kpd * kpd);

con_AUUBH = AUUBH * GeV2nb * jcob;
con_BUUBH = BUUBH * GeV2nb * jcob;

bhAUU = (Gamma/t) * con_AUUBH * ( F1 * F1 + tau * F2 * F2 );
bhBUU = (Gamma/t) * con_BUUBH * ( tau * ( F1 + F2 ) * ( F1 + F2 ) ) ;

xbhUU = bhAUU + bhBUU;
//

AUUI = -4. * cos( phi * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpP + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kP + kpqp * kP_T + kqp * kpP_T - 2.*kkp * kP_T ) -
                  Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * Pqp + 2. * kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) ;

  BUUI = -2. * xi * cos( phi * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpd + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kd + kpqp * kd_T + kqp * kpd_T - 2.*kkp * kd_T ) -
                      Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * qpd + 2. * kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) );

  CUUI = -2. * cos( phi * RAD ) * ( Dplus * ( 2. * kkp * kd_T - kpqp * kd_T - kqp * kpd_T + 4. * xi * kkp * kP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) -
                 Dminus * ( kkp * qpd_T - kpqp * kd_T - kqp * kpd_T + 2. * xi * kkp * qpP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) );

  // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
  con_AUUI = AUUI * GeV2nb * jcob;
  con_BUUI = BUUI * GeV2nb * jcob;
  con_CUUI = CUUI * GeV2nb * jcob;

  //Unpolarized Coefficients multiplied by the Form Factors
  iAUU = (Gamma/(TMath::Abs(t) * QQ)) * con_AUUI * ( F1 * p0 + tau * F2 * p1 );
  iBUU = (Gamma/(TMath::Abs(t) * QQ)) * con_BUUI * ( F1 + F2 ) * ( p0 + p1 );
  iCUU = (Gamma/(TMath::Abs(t) * QQ)) * con_CUUI * ( F1 + F2 ) * p2;

  // Unpolarized BH-DVCS interference cross section
  xIUU = iAUU + iBUU + iCUU;

function = -1.*xIUU + xbhUU + dvcs ;

return function;

  
    
}



void plotting()
{

 //gStyle->SetCanvasColor(0);
 //gStyle->SetOptStat(0);
 gStyle->SetOptFit();


//variable
 bool draw = true;
 int num_hist = 15;
 int nbin = 360;

 double temp[200070];
 float data_k[342];
 float data_QQ[342];
 float data_xb[342];
 float data_t[342];
 float data_F1[342];
 float data_F2[342];
 float data_dvcs[342];
 double data_phi;
 double data_phi_rad;
 double data_F;
 double data_errF;

 TCanvas *cn[342];
 char name[100];
 char name1[100];
 char name2[100];
 char name3[100];
 char name4[100];
 double pi = 3.141592;

 double F[342][45];
 double F_iuu[342][45];
 double F_err[342][45];
 double phi[342][45];
 double F_fit_data[342][45];
 double F_fit_truth[342][45];
 double loss_data[342];
 double loss_truth[342];
 double chisq[342];
 double truth[342][3];
 char name_truth[100];
 double temp_truth[2025];

//
TH1D *ReE_z = new TH1D("ReE_z","ReE_z",105,0,105);
ReE_z->Sumw2();

TH1D *ReH_z = new TH1D("ReH_z","ReH_z",105,0,105);
ReH_z->Sumw2();

TH1D *ReHTilde_z = new TH1D("ReHTilde_z","ReHTilde_z",105,0,105);
ReHTilde_z->Sumw2();
//
 ifstream inputfile_truth;
 sprintf(name_truth,"truth.csv"); 
 inputfile_truth.open(name_truth,ios::in);
 cout<<" Get the data from "<<name_truth<<endl<<endl;
  
 for (int i = 0; i < 2025 ; i++) 
  {
    inputfile_truth >> temp_truth[i];
  } 
 for (int i=0; i<num_hist; i++)
  {
    truth[i][0] = temp_truth[135*i+0];
    truth[i][1] = temp_truth[135*i+1];
    truth[i][2] = temp_truth[135*i+2];
    cerr<<truth[i][0]<<endl;
  }

 TFile f ("testdata.root", "RECREATE", "Histograms from ntuples" ); 

//get the data from text file 
 ifstream inputfile;
 sprintf(name,"dvcs_May2021.txt"); 
 inputfile.open(name,ios::in);
 cout<<" Get the data from "<<name<<endl<<endl;
  
 for (int i = 0; i < 200070 ; i++) 
  {
    inputfile >> temp[i];
  } 

//define the histograms - filling the histogram - write the histograms
// gStyle->SetOptFit(1);
 TH1D *hist_F[342];


 for (int i=0; i<num_hist; i++)
  {
   data_k[i]= temp[i*585 + 2];
   data_QQ[i]= temp[i*585 + 3];
   data_xb[i]= temp[i*585 + 4];
   data_t[i]= temp[i*585 + 5];
   data_F1[i] = temp[i*585 + 10];
   data_F2[i] = temp[i*585 + 11];
   data_dvcs[i] = temp[i*585 + 12];
   sprintf(name1,"set %d, k=%f, QQ=%f, xb=%f, t=%f",(i+1),data_k[i],data_QQ[i],data_xb[i],data_t[i]);
   cerr<<name1<<endl; 
   hist_F[i] = new TH1D(name1,name1, nbin,0,360);
   hist_F[i]->Sumw2();
   hist_F[i]->SetMarkerColor(kBlack);
   hist_F[i]->SetMarkerStyle(kFullCircle);
   hist_F[i]->SetLineColor(kBlack);
   hist_F[i]->SetStats(0);

  for (int j=0; j<45; j++)
   {
    data_phi = temp[i*585+j*13+6];
    data_F = temp[i*585+j*13+7];
    data_errF = temp[i*585+j*13+8];
    data_phi_rad = (data_phi/180)*pi;
    double bin_space = 360.0/nbin;
    int bin_phi = (data_phi-0)/bin_space + 1;
    
    F[i][j] = data_F;
    F_err[i][j] = data_errF;
    phi[i][j] = data_phi;

/*
    Double_t par[3] = {truth[i][0], truth[i][1], truth[i][2]}; 
    double F_interference = data_F - (fitFunction2(&phi[i][j], par));
    
    
    hist_F[i]->SetBinContent(bin_phi, F_interference);
    hist_F[i]->SetBinError(bin_phi, 0.05*F_interference);  
*/
   }
 
  hist_F[i]->Write();
 
 }

 double temp2[45000];
 ifstream inputfile2;
 sprintf(name2,"result26.txt"); 
 inputfile2.open(name2,ios::in);
 cout<<" Get the data from "<<name<<endl<<endl;
  
 for (int i = 0; i < 45000 ; i++) 
  {
    inputfile2 >> temp2[i];
  } 

 TH1D *bin_zul_ReH[342];
 char namehisto3[100];
 for (int i=0; i<15; i++)
  {
   sprintf(namehisto3,"histo_ReH_zul_%i",i);
   bin_zul_ReH[i] = new TH1D(namehisto3, namehisto3,100, -10,10);
   bin_zul_ReH[i]->Sumw2();

   for (int j=0; j<1000; j++)
    {
      bin_zul_ReH[i]->Fill(temp2[15*3*j+3*i+0]);
    }

   bin_zul_ReH[i]->Fit("gaus");
   TF1 *gfit = (TF1 *)bin_zul_ReH[i]->GetFunction("gaus");
   ReH_fit[i] = gfit->GetParameter(1);
   ReH_err[i] = gfit->GetParameter(2);
   
   ReH_z->SetBinContent(7*i+1,ReH_fit[i]);
   ReH_z->SetBinError(7*i+1,ReH_err[i]);
  }

 TH1D *bin_zul_ReE[342];
 char namehisto4[100];
 for (int i=0; i<15; i++)
  {
   sprintf(namehisto4,"histo_ReE_zul_%i",i);
   bin_zul_ReE[i] = new TH1D(namehisto4, namehisto4,100, -10,10);
   bin_zul_ReE[i]->Sumw2();

   for (int j=0; j<1000; j++)
    {
      bin_zul_ReE[i]->Fill(temp2[15*3*j+3*i+1]);
    }

   bin_zul_ReE[i]->Fit("gaus");
   TF1 *gfit = (TF1 *)bin_zul_ReE[i]->GetFunction("gaus");
   ReE_fit[i] = gfit->GetParameter(1);
   ReE_err[i] = gfit->GetParameter(2);

   ReE_z->SetBinContent(7*i+1,ReE_fit[i]);
   ReE_z->SetBinError(7*i+1,ReE_err[i]);
  }

 TH1D *bin_zul_ReHTilde[342];
 char namehisto5[100];
 for (int i=0; i<15; i++)
  {
   sprintf(namehisto5,"histo_ReHTilde_zul_%i",i);
   bin_zul_ReHTilde[i] = new TH1D(namehisto5, namehisto5,100, -10,10);
   bin_zul_ReHTilde[i]->Sumw2();

   for (int j=0; j<1000; j++)
    {
      bin_zul_ReHTilde[i]->Fill(temp2[15*3*j+3*i+2]);
    }

   bin_zul_ReHTilde[i]->Fit("gaus");
   TF1 *gfit = (TF1 *)bin_zul_ReHTilde[i]->GetFunction("gaus");
   ReHTilde_fit[i] = gfit->GetParameter(1);
   ReHTilde_err[i] = gfit->GetParameter(2);

   ReHTilde_z->SetBinContent(7*i+1,ReHTilde_fit[i]);
   ReHTilde_z->SetBinError(7*i+1,ReHTilde_err[i]);
  }

  
 
 for (int i=0; i<num_hist; i++)
 {

  k = data_k[i];
  QQ = data_QQ[i];
  xb = data_xb[i];
  t= data_t[i];
  F1 = data_F1[i];
  F2 = data_F2[i];
  dvcs = data_dvcs[i];

  loss_data[i] = 0.;
  loss_truth[i] = 0.;
  chisq[i] = 0;


  Double_t kins_data[3] = {ReH_fit[i], ReE_fit[i], ReHTilde_fit[i]};
  Double_t kins[3] = {truth[i][0], truth[i][1], truth[i][2]}; 
//
  
  for (int j=0; j<45; j++)
      {
       
       F_fit_data[i][j] = fitFunction(&phi[i][j], kins_data);
       F_iuu[i][j] = F[i][j] - (fitFunction2(&phi[i][j], kins));
       //loss_data[i] = loss_data[i] + fabs((F[i][j] - F_fit_data[i][j])/F_iuu[i][j]);
       loss_data[i] = loss_data[i] + (F[i][j] - F_fit_data[i][j])*(F[i][j] - F_fit_data[i][j])/(F_err[i][j]*F_err[i][j]);
       
      }
  //loss_data[i] = loss[i]/45.;
  
//

//
  
  for (int j=0; j<45; j++)
      {
       
       F_fit_truth[i][j] = fitFunction(&phi[i][j], kins);
       F_iuu[i][j] = F[i][j] - (fitFunction2(&phi[i][j], kins));
       //loss_truth[i] = loss_truth[i] + fabs((F[i][j] - F_fit_truth[i][j])/F_iuu[i][j]);
       loss_truth[i] = loss_truth[i] + (F[i][j] - F_fit_truth[i][j])*(F[i][j] - F_fit_truth[i][j])/(F_err[i][j]*F_err[i][j]);
       
      }
  //loss_truth[i] = loss[i]/45.;
  
//
  sprintf(name3,"canvas%i",i);
  sprintf(name4,"plotv2e_result26_canvas%i_.png",i);
  cn[i] = new TCanvas(name3,name3);
  cn[i]->cd();


  TF1 *fitLine = new TF1("fitLine",fitFunction3,0,360,3);
  fitLine->SetLineWidth(3);
  fitLine->SetLineColor(kRed);
  fitLine->FixParameter(0,ReH_fit[i]);
  fitLine->FixParameter(1,ReE_fit[i]);
  fitLine->FixParameter(2,ReHTilde_fit[i]);

  TF1 *fitLine2 = new TF1("fitLine2",fitFunction3,0,360,3);
  fitLine2->SetLineWidth(3);
  fitLine2->SetLineColor(kBlue);
  fitLine2->FixParameter(0,truth[i][0]);
  fitLine2->FixParameter(1,truth[i][1]);
  fitLine2->FixParameter(2,truth[i][2]);
  cerr<<"set: "<<(i+1)<<" "<<ReH_fit[i]<<" "<<ReE_fit[i]<<" "<<ReHTilde_fit[i]<<"  "<<"loss data = "<<loss_data[i]<<endl;
  cerr<<"set: "<<(i+1)<<" "<<truth[i][0]<<" "<<truth[i][1]<<" "<<truth[i][2]<<"  "<<"loss truth = "<<loss_truth[i]<<endl;

  hist_F[i]->GetXaxis()->SetTitle("#phi_{x}");
  hist_F[i]->GetYaxis()->SetTitle("F_iuu");
  

 //fitLine->Draw();
 //fitLine2->Draw("SAME");

 for (int j=0; j<45; j++)
 {
  Double_t par[3] = {truth[i][0], truth[i][1], truth[i][2]}; 
  double F_interference = F[i][j] - (fitFunction2(&phi[i][j], par));
    double bin_space = 360.0/360;
    int bin_phi = (phi[i][j]-0)/bin_space + 1;
    
    hist_F[i]->SetBinContent(bin_phi, F_interference);
    hist_F[i]->SetBinError(bin_phi, F_err[i][j]);  
 }
 hist_F[i]->Draw("e");
 fitLine->Draw("SAME");
 fitLine2->Draw("SAME");

// bin_zul_ReHTilde[i]->Draw();
 cn[i]->SaveAs(name4);
   
}



TCanvas *d0 = new TCanvas("d0","d0"); //c0->SetGrid();
	d0->cd();

			
	ReE_z->SetTitle("ReE ; Set; ");
	ReE_z->SetMarkerColor(kRed);
	ReE_z->SetMarkerStyle(kFullCircle);
	ReE_z->SetLineColor(kRed);
	ReE_z->SetStats(0);
	ReE_z->SetMaximum(10);
	ReE_z->SetMinimum(-10);
	ReE_z->Draw("E1 ");

	for (int i=0; i<15; i++)
	{
	  TLine *line = new TLine(i*7,truth[i][1],i*7+5,truth[i][1]);
	  line->Draw("SAME");
	  //line->SetLineStyle(7);
	}

TCanvas *d1 = new TCanvas("d1","d1"); //c0->SetGrid();
	d1->cd();

	ReH_z->SetTitle("ReH ; Set; ");
	ReH_z->SetMarkerColor(kRed);
	ReH_z->SetMarkerStyle(kFullCircle);
	ReH_z->SetLineColor(kRed);
	ReH_z->SetStats(0);
	ReH_z->SetMaximum(10);
	ReH_z->SetMinimum(-10);
	ReH_z->Draw("E1 ");

	for (int i=0; i<15; i++)
	{
	  TLine *line = new TLine(i*7,truth[i][0],i*7+5,truth[i][0]);
	  line->Draw("SAME");
	  //line->SetLineStyle(7);
  	}

TCanvas *d2 = new TCanvas("d2","d2"); //c0->SetGrid();
	d2->cd();

	ReHTilde_z->SetTitle("ReHTilde ; Set; ");
	ReHTilde_z->SetMarkerColor(kRed);
	ReHTilde_z->SetMarkerStyle(kFullCircle);
	ReHTilde_z->SetLineColor(kRed);
	ReHTilde_z->SetStats(0);
	ReHTilde_z->SetMaximum(10);
	ReHTilde_z->SetMinimum(-10);
	ReHTilde_z->Draw("E1 ");

	for (int i=0; i<15; i++)
	{
	  TLine *line = new TLine(i*7,truth[i][2],i*7+5,truth[i][2]);
	  line->Draw("SAME");
	  //line->SetLineStyle(7);
  	}



}    

Double_t fitFunction2(Double_t *x, Double_t *par) 
{

const double p0 = par[0];
const double p1 = par[1];
const double p2 = par[2];
const double phi = x[0];



Double_t ALP_INV = 137.0359998; 
Double_t PI = 3.1415926535;
Double_t RAD = PI / 180.;
Double_t M = 0.938272; 
Double_t GeV2nb = .389379*1000000; 

Double_t chisq = 0.0;
Double_t function = 0.0;
Double_t prob = 0.0;
Double_t nll = 0.0;


Double_t y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau;

  
TLorentzVector K, KP, Q, QP, D, p, P;
Double_t kkp, kq, kp, kpp;
Double_t kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd;
Double_t kk_T, kqp_T, kkp_T, kd_T, dd_T, kpqp_T, kP_T, kpP_T, qpP_T, kpd_T, qpd_T, Dplus, Dminus;

Double_t s;     
Double_t Gamma; 
Double_t jcob;  
Double_t AUUBH, BUUBH; 
Double_t AUUI, BUUI, CUUI; 
Double_t con_AUUBH, con_BUUBH, con_AUUI, con_BUUI, con_CUUI;  
Double_t bhAUU, bhBUU; 
Double_t iAUU, iBUU, iCUU; 
Double_t real_iAUU, real_iBUU, real_iCUU; 
Double_t xbhUU;
Double_t xIUU; 
Double_t real_xbhUU;
Double_t real_xIUU; 




M2 = M*M; 
y = QQ / ( 2. * M * k * xb ); 
  
gg = 4. * M2 * xb * xb / QQ; 
e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); 
xi = 1. * xb * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - xb + xb * t / QQ ) ); 
tmin = -( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( xb * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* xb ) ) ); 
kpr = k * ( 1. - y ); 

qp = t / 2. / M + k - kpr; 
po = M - t / 2. / M; 
pmag = sqrt( ( -t ) * ( 1. - t / 4. / M / M ) ); 
cth = -1. / sqrt( 1. + gg ) * ( 1. + gg / 2. * ( 1. + t / QQ ) / ( 1. + xb * t / QQ ) ); 
theta = acos(cth); 
sthl = sqrt( gg ) / sqrt( 1. + gg ) * ( sqrt ( 1. - y - y * y * gg / 4. ) ); 
cthl = -1. / sqrt( 1. + gg ) * ( 1. + y * gg / 2. ) ; 
tau = -0.25 * t / M2;
K.SetPxPyPzE( k * sthl, 0.0, k * cthl, k );
KP.SetPxPyPzE( K(0), 0.0, k * ( cthl + y * sqrt( 1. + gg ) ), kpr );
Q = K - KP;
p.SetPxPyPzE(0.0, 0.0, 0.0, M);

s = (p + K) * (p + K);
Gamma = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / ( s - M2 ) / ( s - M2 ) / sqrt( 1. + gg ) / xb;
jcob = 2. * PI ;

//
QP.SetPxPyPzE(qp * sin(theta) * cos( RAD*phi ), qp * sin(theta) * sin( RAD*phi ), qp * cos(theta), qp);
D = Q - QP; 
TLorentzVector pp = p + D; 
P = p + pp;
P.SetPxPyPzE(.5*P.Px(), .5*P.Py(), .5*P.Pz(), .5*P.E());

//
kkp  = K * KP;   
kq   = K * Q;    
kp   = K * p;    
kpp  = KP * p;   

kd   = K * D;  
kpd  = KP * D;   
kP   = K * P;    
kpP  = KP * P;   
kqp  = K * QP;   
kpqp = KP * QP;  
dd   = D * D;    
Pq   = P * Q;    
Pqp  = P * QP;   
qd   = Q * D;    
qpd  = QP * D;   

//

kk_T   = TProduct(K,K);
  kkp_T  = kk_T;
  kqp_T  = TProduct(K,QP);
  kd_T   = -1.* kqp_T;
  dd_T   = TProduct(D,D);
  kpqp_T = kqp_T;
  kP_T   = TProduct(K,P);
  kpP_T  = TProduct(KP,P);
  qpP_T  = TProduct(QP,P);
  kpd_T  = -1.* kqp_T;
  qpd_T  = -1. * dd_T;

  Dplus   = .5 / kpqp - .5 / kqp;
  Dminus  = -.5 / kpqp - .5 / kqp;


//
AUUBH = ( (8. * M2) / (t * kqp * kpqp) ) * ( (4. * tau * (kP * kP + kpP * kpP) ) - ( (tau + 1.) * (kd * kd + kpd * kpd) ) );
BUUBH = ( (16. * M2) / (t* kqp * kpqp) ) * (kd * kd + kpd * kpd);

con_AUUBH = AUUBH * GeV2nb * jcob;
con_BUUBH = BUUBH * GeV2nb * jcob;

bhAUU = (Gamma/t) * con_AUUBH * ( F1 * F1 + tau * F2 * F2 );
bhBUU = (Gamma/t) * con_BUUBH * ( tau * ( F1 + F2 ) * ( F1 + F2 ) ) ;

xbhUU = bhAUU + bhBUU;
//

AUUI = -4. * cos( phi * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpP + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kP + kpqp * kP_T + kqp * kpP_T - 2.*kkp * kP_T ) -
                  Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * Pqp + 2. * kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) ;

  BUUI = -2. * xi * cos( phi * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpd + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kd + kpqp * kd_T + kqp * kpd_T - 2.*kkp * kd_T ) -
                      Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * qpd + 2. * kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) );

  CUUI = -2. * cos( phi * RAD ) * ( Dplus * ( 2. * kkp * kd_T - kpqp * kd_T - kqp * kpd_T + 4. * xi * kkp * kP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) -
                 Dminus * ( kkp * qpd_T - kpqp * kd_T - kqp * kpd_T + 2. * xi * kkp * qpP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) );

  // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
  con_AUUI = AUUI * GeV2nb * jcob;
  con_BUUI = BUUI * GeV2nb * jcob;
  con_CUUI = CUUI * GeV2nb * jcob;

  //Unpolarized Coefficients multiplied by the Form Factors
  iAUU = (Gamma/(TMath::Abs(t) * QQ)) * con_AUUI * ( F1 * p0 + tau * F2 * p1 );
  iBUU = (Gamma/(TMath::Abs(t) * QQ)) * con_BUUI * ( F1 + F2 ) * ( p0 + p1 );
  iCUU = (Gamma/(TMath::Abs(t) * QQ)) * con_CUUI * ( F1 + F2 ) * p2;

  // Unpolarized BH-DVCS interference cross section
  xIUU = iAUU + iBUU + iCUU;

function = xbhUU + dvcs ;

return function;
     
}

Double_t fitFunction3(Double_t *x, Double_t *par) 
{

const double p0 = par[0];
const double p1 = par[1];
const double p2 = par[2];
const double phi = x[0];



Double_t ALP_INV = 137.0359998; 
Double_t PI = 3.1415926535;
Double_t RAD = PI / 180.;
Double_t M = 0.938272; 
Double_t GeV2nb = .389379*1000000; 

Double_t chisq = 0.0;
Double_t function = 0.0;
Double_t prob = 0.0;
Double_t nll = 0.0;


Double_t y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau;

  
TLorentzVector K, KP, Q, QP, D, p, P;
Double_t kkp, kq, kp, kpp;
Double_t kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd;
Double_t kk_T, kqp_T, kkp_T, kd_T, dd_T, kpqp_T, kP_T, kpP_T, qpP_T, kpd_T, qpd_T, Dplus, Dminus;

Double_t s;     
Double_t Gamma; 
Double_t jcob;  
Double_t AUUBH, BUUBH; 
Double_t AUUI, BUUI, CUUI; 
Double_t con_AUUBH, con_BUUBH, con_AUUI, con_BUUI, con_CUUI;  
Double_t bhAUU, bhBUU; 
Double_t iAUU, iBUU, iCUU; 
Double_t real_iAUU, real_iBUU, real_iCUU; 
Double_t xbhUU;
Double_t xIUU; 
Double_t real_xbhUU;
Double_t real_xIUU; 




M2 = M*M; 
y = QQ / ( 2. * M * k * xb ); 
  
gg = 4. * M2 * xb * xb / QQ; 
e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); 
xi = 1. * xb * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - xb + xb * t / QQ ) ); 
tmin = -( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( xb * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* xb ) ) ); 
kpr = k * ( 1. - y ); 

qp = t / 2. / M + k - kpr; 
po = M - t / 2. / M; 
pmag = sqrt( ( -t ) * ( 1. - t / 4. / M / M ) ); 
cth = -1. / sqrt( 1. + gg ) * ( 1. + gg / 2. * ( 1. + t / QQ ) / ( 1. + xb * t / QQ ) ); 
theta = acos(cth); 
sthl = sqrt( gg ) / sqrt( 1. + gg ) * ( sqrt ( 1. - y - y * y * gg / 4. ) ); 
cthl = -1. / sqrt( 1. + gg ) * ( 1. + y * gg / 2. ) ; 
tau = -0.25 * t / M2;
K.SetPxPyPzE( k * sthl, 0.0, k * cthl, k );
KP.SetPxPyPzE( K(0), 0.0, k * ( cthl + y * sqrt( 1. + gg ) ), kpr );
Q = K - KP;
p.SetPxPyPzE(0.0, 0.0, 0.0, M);

s = (p + K) * (p + K);
Gamma = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / ( s - M2 ) / ( s - M2 ) / sqrt( 1. + gg ) / xb;
jcob = 2. * PI ;

//
QP.SetPxPyPzE(qp * sin(theta) * cos( RAD*phi ), qp * sin(theta) * sin( RAD*phi ), qp * cos(theta), qp);
D = Q - QP; 
TLorentzVector pp = p + D; 
P = p + pp;
P.SetPxPyPzE(.5*P.Px(), .5*P.Py(), .5*P.Pz(), .5*P.E());

//
kkp  = K * KP;   
kq   = K * Q;    
kp   = K * p;    
kpp  = KP * p;   

kd   = K * D;  
kpd  = KP * D;   
kP   = K * P;    
kpP  = KP * P;   
kqp  = K * QP;   
kpqp = KP * QP;  
dd   = D * D;    
Pq   = P * Q;    
Pqp  = P * QP;   
qd   = Q * D;    
qpd  = QP * D;   

//

kk_T   = TProduct(K,K);
  kkp_T  = kk_T;
  kqp_T  = TProduct(K,QP);
  kd_T   = -1.* kqp_T;
  dd_T   = TProduct(D,D);
  kpqp_T = kqp_T;
  kP_T   = TProduct(K,P);
  kpP_T  = TProduct(KP,P);
  qpP_T  = TProduct(QP,P);
  kpd_T  = -1.* kqp_T;
  qpd_T  = -1. * dd_T;

  Dplus   = .5 / kpqp - .5 / kqp;
  Dminus  = -.5 / kpqp - .5 / kqp;


//
AUUBH = ( (8. * M2) / (t * kqp * kpqp) ) * ( (4. * tau * (kP * kP + kpP * kpP) ) - ( (tau + 1.) * (kd * kd + kpd * kpd) ) );
BUUBH = ( (16. * M2) / (t* kqp * kpqp) ) * (kd * kd + kpd * kpd);

con_AUUBH = AUUBH * GeV2nb * jcob;
con_BUUBH = BUUBH * GeV2nb * jcob;

bhAUU = (Gamma/t) * con_AUUBH * ( F1 * F1 + tau * F2 * F2 );
bhBUU = (Gamma/t) * con_BUUBH * ( tau * ( F1 + F2 ) * ( F1 + F2 ) ) ;

xbhUU = bhAUU + bhBUU;
//

AUUI = -4. * cos( phi * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpP + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kP + kpqp * kP_T + kqp * kpP_T - 2.*kkp * kP_T ) -
                  Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * Pqp + 2. * kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) ;

  BUUI = -2. * xi * cos( phi * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpd + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kd + kpqp * kd_T + kqp * kpd_T - 2.*kkp * kd_T ) -
                      Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * qpd + 2. * kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) );

  CUUI = -2. * cos( phi * RAD ) * ( Dplus * ( 2. * kkp * kd_T - kpqp * kd_T - kqp * kpd_T + 4. * xi * kkp * kP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) -
                 Dminus * ( kkp * qpd_T - kpqp * kd_T - kqp * kpd_T + 2. * xi * kkp * qpP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) );

  // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
  con_AUUI = AUUI * GeV2nb * jcob;
  con_BUUI = BUUI * GeV2nb * jcob;
  con_CUUI = CUUI * GeV2nb * jcob;

  //Unpolarized Coefficients multiplied by the Form Factors
  iAUU = (Gamma/(TMath::Abs(t) * QQ)) * con_AUUI * ( F1 * p0 + tau * F2 * p1 );
  iBUU = (Gamma/(TMath::Abs(t) * QQ)) * con_BUUI * ( F1 + F2 ) * ( p0 + p1 );
  iCUU = (Gamma/(TMath::Abs(t) * QQ)) * con_CUUI * ( F1 + F2 ) * p2;

  // Unpolarized BH-DVCS interference cross section
  xIUU = iAUU + iBUU + iCUU;

function = -1.*xIUU ;

return function;
     
} 
