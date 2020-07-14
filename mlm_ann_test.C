#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include "TFile.h"
#include "TPostScript.h"
#include <iomanip>
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TLorentzRotation.h"
#include "TVector3.h"
#include "TMath.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TGaxis.h"
#include "TLatex.h"
#include "TMarker.h"
#include "TPaveText.h"

#include "TMinuit.h"
#include "TLegend.h"
#include "TText.h"
#include "TMatrix.h"
#include "TMath.h"



using namespace std; 


Double_t data_QQ[100];
Double_t data_xb[100];
Double_t data_t[100];
Double_t data_k[100];
Double_t data_F1[100];
Double_t data_F2[100];
Double_t data_F[100];
Double_t data_err_F[100];
Double_t data_phi[100];
Double_t data_dvcs[100];

int jj;
double temp[50000];


void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t par[], Int_t iflag);


void mlm_ann_test()
{

TCanvas *cn[89];
TCanvas *cn1[100];
TCanvas *cn2[100];
char name[100];
char name1[100];
char name2[100];
char name3[100];
char name4[100],name5[100],name6[100],name7[100],name8[100];

//double temp[50000];
double pi = 3.141592;
bool draw = true;
int num_hist = 15;

//TFile f ("testdata.root", "RECREATE", "Histograms from ntuples" ); 

//get the data from text file which is data1.txt which is copied from dvcs_cross_fixed_t.csv
ifstream inputfile;
sprintf(name,"data_july7.txt"); 
inputfile.open(name,ios::in);
cout<<" Get the data from "<<name<<endl<<endl;
  
for (int i = 0; i < 8100 ; i++) 
 {
  inputfile >> temp[i];
 } 

//define the histograms - filling the histogram - write the histograms
TH1D *h_ReE = new TH1D("ReE","ReE",15,0,15);
TH1D *h_ReH = new TH1D("ReH","ReH",15,0,15);
TH1D *h_ReHTilde = new TH1D("ReHTilde","ReHTilde",15,0,15);


//Minuit Fitting
for (int jjj=0; jjj<num_hist; jjj++)
{
jj = jjj;
cerr<<"START FITTING HISTOGRAM: "<<jj<<" "<<data_k[jj]<<" "<<data_t[jj]<<endl;
int flag;
gMinuit = new TMinuit(3); 
gMinuit -> SetPrintLevel(0); 
gMinuit->SetFCN(fcn);

//double par_init[3] = {temp[jjj*540 + 13]-0.5,temp[jjj*540 + 12]+0.5,temp[jjj*540 + 14]-0.5};
//double par_init[3] = {1.0,1.0,1.0 };
double par_init[3] = {(rand() % 200) - 100,(rand() % 200) - 100,(rand() % 200) - 100 };
double step_size = /*0.0025*/0.0025;
double upper_bound[3] = {100.0,100.0,100.0};
double lower_bound[3] = {-100.0,-100.0,-100.0};

gMinuit->mnparm(0,"par_0",par_init[0],step_size,lower_bound[0],upper_bound[0],flag);
gMinuit->mnparm(1,"par_1",par_init[1],step_size,lower_bound[1],upper_bound[1],flag);
gMinuit->mnparm(2,"par_2",par_init[2],step_size,lower_bound[2],upper_bound[2],flag);

/*gMinuit->mnparm(0,"par_0",0.001,step_size,0.0,0.0,flag);
gMinuit->mnparm(1,"par_1",0.001,step_size,0.0,0.0,flag);
gMinuit->mnparm(2,"par_2",0.001,step_size,0.0,0.0,flag);*/
 

//Improvement algorithm
Double_t arglist[10];
Int_t ierflg = 0;
arglist[0] = 25000;
arglist[1] = 1.;
//gMinuit->mnexcm("MINIMIZE", arglist ,2,ierflg);

gMinuit->Migrad();
//gMinuit->Minimize();



double parreturn[3], parreturnError[3];
    
for(int parn = 0; parn < 3; parn++)
 {
  gMinuit->GetParameter(parn,parreturn[parn],parreturnError[parn]);
 }

cerr<<"END FITTING HISTOGRAM: "<<jj<<" "<<(temp[jjj*540 + 13]-parreturn[0])/temp[jjj*540 + 13]<<" "<<(temp[jjj*540 + 12]-parreturn[1])/temp[jjj*540 + 12]<<" "<<(temp[jjj*540 + 14]-parreturn[2])/temp[jjj*540 + 14]<<endl;

if (fabs(parreturn[0]) < 1000)
{
h_ReE->SetBinContent(jjj+1,(temp[jjj*540 + 13]-parreturn[0])/temp[jjj*540 + 13]);
h_ReE->SetBinError(jjj+1,0.001);
}
if (fabs(parreturn[1]) < 1000)
{
h_ReH->SetBinContent(jjj+1,(temp[jjj*540 + 12]-parreturn[1])/temp[jjj*540 + 12]);
h_ReH->SetBinError(jjj+1,0.001);
}
if (fabs(parreturn[2]) < 1000)
{
h_ReHTilde->SetBinContent(jjj+1,(temp[jjj*540 + 14]-parreturn[2])/temp[jjj*540 + 14]);
h_ReHTilde->SetBinError(jjj+1,0.001);
}

}

i=0;
sprintf(name5,"canvas5%i",i);
sprintf(name6,"mlm_ReE4.png",i);
cn[i] = new TCanvas(name5,name5);
cn[i]->cd();
//h_ReE->SetLineColor(kRed);
h_ReE->GetXaxis()->SetTitle("Hist");
h_ReE->GetYaxis()->SetTitle("ReE");
h_ReE->SetMaximum(1);
h_ReE->SetMinimum(-1);
h_ReE->SetMarkerStyle(20);
h_ReE->SetMarkerColor(kRed);
h_ReE->SetMarkerSize(1);
h_ReE->Draw();
cn[i]->SaveAs(name6);

sprintf(name5,"canvas6%i",i);
sprintf(name6,"mlm_ReH4.png",i);
cn1[i] = new TCanvas(name5,name5);
cn1[i]->cd();
//h_ReH->SetLineColor(kRed);
h_ReH->GetXaxis()->SetTitle("Hist");
h_ReH->GetYaxis()->SetTitle("ReH");
h_ReH->SetMaximum(1);
h_ReH->SetMinimum(-1);
h_ReH->SetMarkerStyle(20);
h_ReH->SetMarkerColor(kRed);
h_ReH->SetMarkerSize(1);
h_ReH->Draw();
cn1[i]->SaveAs(name6);

sprintf(name5,"canvas7%i",i);
sprintf(name6,"mlm_ReHTilde4.png",i);
cn2[i] = new TCanvas(name5,name5);
cn2[i]->cd();
//h_ReHTilde->SetLineColor(kRed);
h_ReHTilde->GetXaxis()->SetTitle("Hist");
h_ReHTilde->GetYaxis()->SetTitle("ReHTilde");
h_ReHTilde->SetMaximum(1);
h_ReHTilde->SetMinimum(-1);
h_ReHTilde->SetMarkerStyle(20);
h_ReHTilde->SetMarkerColor(kRed);
h_ReHTilde->SetMarkerSize(1);
h_ReHTilde->Draw();
cn2[i]->SaveAs(name6);
}


void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t par[], Int_t iflag)
{
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
Double_t kk_T, kqp_T, kkp_T, kd_T, dd_T;

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

Double_t k, QQ, xb, t, phi, data, sigma, F1, F2, dvcs, w, norm, sigma_test, real_function;
Double_t real_par0, real_par1, real_par2;
Double_t norm_data, norm_function;
norm_data =0.0;
norm_function = 0.0;

for (int i=0; i<36; i++)
{
  norm_data = norm_data + temp[jj*540+i*15+7];
}



for (int i=0; i<36; i++)
{
k = temp[jj*540+i*15+2];
QQ = temp[jj*540+i*15+3];
xb = temp[jj*540+i*15+4];
t = temp[jj*540+i*15+5];

phi = temp[jj*540+i*15+6];
data = temp[jj*540+i*15+7];
sigma = temp[jj*540+i*15+8];

F1 = temp[jj*540+i*15+9];
F2 = temp[jj*540+i*15+10];
dvcs = temp[jj*540+i*15+11];

real_par0 = temp[jj*540+i*15+13];
real_par1 = temp[jj*540+i*15+12];
real_par2 = temp[jj*540+i*15+14];

sigma_test = 0.1*data;
//w = 1.0/sigma;
w = 1.0;
norm = 10000.0;
//norm_data = norm_data + data;


M2 = M*M; 
y = QQ / ( 2. * M * k * xb ); 
  
gg = 4. * M2 * xb * xb / QQ; 
e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); 
xi = 1. * xb * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - xb + xb * t / QQ ) ); 
tmin = ( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( xb * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* xb ) ) ); 
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
jcob = 1./ ( 2. * M * xb * K(3) ) * 2. * PI * 2.;

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
kk_T = 0.5 * ( e / ( 1 - e ) ) * QQ;  
kkp_T = kk_T;  
kqp_T = ( QQ / ( sqrt( gg ) * sqrt( 1 + gg ) ) ) * sqrt ( (0.5 * e) / ( 1 - e ) ) * ( 1. + xb * t / QQ ) * sin(theta) * cos( RAD*phi );
kd_T = -1.* kqp_T;
dd_T = ( 1. - xi * xi ) * ( tmin - t );

//
AUUBH = ( (8. * M2) / (t * kqp * kpqp) ) * ( (4. * tau * (kP * kP + kpP * kpP) ) - ( (tau + 1.) * (kd * kd + kpd * kpd) ) );
BUUBH = ( (16. * M2) / (t* kqp * kpqp) ) * (kd * kd + kpd * kpd);

con_AUUBH = 2. * AUUBH * GeV2nb * jcob;
con_BUUBH = 2. * BUUBH * GeV2nb * jcob;

bhAUU = (Gamma/t) * con_AUUBH * ( F1 * F1 + tau * F2 * F2 );
bhBUU = (Gamma/t) * con_BUUBH * ( tau * ( F1 + F2 ) * ( F1 + F2 ) ) ;

xbhUU = bhAUU + bhBUU;
//
AUUI = -4.0 * cos( RAD*phi ) / (kqp * kpqp) * ( ( QQ + t ) * ( 2.0 * ( kP + kpP ) * kk_T   + ( Pq * kqp_T ) + 2.* ( kpP * kqp ) - 2.* ( kP * kpqp ) ) + ( QQ - t + 4.* kd ) * Pqp * ( kkp_T + kqp_T - 2.* kkp ) );
BUUI = 2.0 * xi * cos( RAD*phi ) / ( kqp * kpqp) * ( ( QQ + t ) * ( 2.* kk_T * ( kd + kpd ) + kqp_T * ( qd - kqp - kpqp + 2.*kkp ) + 2.* kqp * kpd - 2.* kpqp * kd ) + ( QQ - t + 4.* kd ) * ( ( kk_T - 2.* kkp ) * qpd - kkp * dd_T - 2.* kd_T * kqp ) ) / tau;
CUUI = 2.0 * cos( RAD*phi ) / ( kqp * kpqp) * ( -1. * ( QQ + t ) * ( 2.* kkp - kpqp - kqp ) * kd_T + ( QQ - t + 4.* kd ) * ( ( kqp + kpqp ) * kd_T + dd_T * kkp ) );

con_AUUI = AUUI * GeV2nb * jcob;
con_BUUI = BUUI * GeV2nb * jcob;
con_CUUI = CUUI * GeV2nb * jcob;

iAUU = (Gamma/(-t * QQ)) * con_AUUI * ( F1 * par[1] + tau * F2 * par[0] );
iBUU = (Gamma/(-t * QQ)) * con_BUUI * tau * ( F1 + F2 ) * ( par[1] + par[0] );
iCUU = (Gamma/(-t * QQ)) * con_CUUI * ( F1 + F2 ) * par[2];

real_iAUU = (Gamma/(-t * QQ)) * con_AUUI * ( F1 * real_par1 + tau * F2 * real_par0 );
real_iBUU = (Gamma/(-t * QQ)) * con_BUUI * tau * ( F1 + F2 ) * ( real_par1 + real_par0 );
real_iCUU = (Gamma/(-t * QQ)) * con_CUUI * ( F1 + F2 ) * real_par2;

xIUU = iAUU + iBUU + iCUU ;
real_xIUU = real_iAUU + real_iBUU + real_iCUU ;

function = xIUU + xbhUU + dvcs ;
real_function = real_xIUU + xbhUU + dvcs ;

//nll = nll + (data - function)*(data - function)/(sigma*sigma); // chi squarefit
nll = nll + function - data + data*TMath::Log(data/function); //main mlm function
//nll = nll + function - data*TMath::Log(function); //test function


}



f = 2*nll;

}    
