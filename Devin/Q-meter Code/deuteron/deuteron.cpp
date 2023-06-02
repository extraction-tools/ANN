#include "TF1.h"
#include "TMath.h"
#include "TCanvas.h"
#include <iostream>
#include "TROOT.h"
#include "TApplication.h"
#include "TGClient.h"

using namespace std;

double gamm = 0.05;
double epsilon = 1;
double ncos2phi = 0.04;

double pi = TMath::Pi();

double F(double *r, double *par) {
  (void)par;
  double R = r[0];
  double x2 = TMath::Sqrt(gamm*gamm + TMath::Power(1-(epsilon*R)-ncos2phi,2));
  double x = TMath::Sqrt(x2);
  double y = TMath::Sqrt(3-ncos2phi);
  double y2 = y*y;
  double cos_alpha = (1-epsilon*R-ncos2phi)/x2;
  double alpha = TMath::ACos(cos_alpha);

  double term1 = 2*TMath::Cos(alpha/2)*(TMath::ATan((y2-x2)/(2*y*x*TMath::Sin(alpha/2)))+(pi/2));
  double subterm1 = y2 + x2 + 2*y*x*TMath::Cos(alpha/2);
  double subterm2 = y2 + x2 - 2*y*x*TMath::Cos(alpha/2);
  double term2 = TMath::Sin(alpha/2)*TMath::Log(subterm1/subterm2);
  return (1/(2*pi*x))*(term1+term2);
}

int main(int argc, char **argv) {
  // ------------------Setting up ROOT Display----------------------------

  TApplication theApp("App", &argc, argv); // init ROOT App for display
  UInt_t dh = gClient->GetDisplayHeight()/2;   // fix plot to 1/2 screen height 
  UInt_t dw = 1.1*dh;
  TCanvas *c1 = new TCanvas("c1","Solutions",dw,dh);
  c1->cd();
  //---------------------------------------------------------------------

  TF1 *f1 = new TF1("myfunc",F,-6,6,0);
  f1->Draw();
  
  FILE *fp = fopen("DEUTERON.DAT","w");
  FILE *fp2 = fopen("DDEUTERON.DAT","w");
  
  double leftbound = -6;
  double rightbound = 6;
  int numpoints = 256;
  double delta = (rightbound-leftbound)/numpoints;
  double h = 0.001;
  double x[1];
  double xplush[1];
  double par[0];
  for (int i = 0; i < numpoints; i++) {
    x[0] = leftbound + i*delta;
    xplush[0] = x[0] + h;
    double val = F(x,par);
    double val2 = (F(xplush,par)-F(x,par))/h;
    fprintf(fp,"%lf",val);
    fprintf(fp2,"%lf",val2);
    if (i != numpoints-1) {
      fprintf(fp,"\n");
      fprintf(fp2,"\n");
    }
  }
  fclose(fp);
  fclose(fp2);

  //---------Drawing display and running the application-----------------
  
  c1->Draw();
  cout << "Press ^c to exit" << endl;
  theApp.SetIdleTimer(300,".q");  // set up a failsafe timer to end the program  
  theApp.Run();
  //--------
}
