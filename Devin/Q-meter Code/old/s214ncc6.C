//Colin Crovella

//Proton at 214 MHz with 7 l/2 cable
//WITH stray capacitance to ground in coil BUT INCLUDE NON-CONSTANT COIL CURRENT

#include <math.h>
#include <stdio.h>
#include <complex.h>

double knob = 0.885;

double U = 0.1;  //100 mV RF input
double L0 = 3*pow(10,-8); //Inductance of coil 30 nH
double Rcoil = 0.35; //Resistance of coil 0.35 ohm
double f = 213*pow(10,6);
double wRes = 2*M_PI*f; //Resonant frequency

double r = 10; //Damping resistor
double R = 619; //619 ohm
double Cmain = 20*pow(10,-12)*knob;
double Cstray = 1*pow(10,-15); //stray capacitance ~ 1 femtofarad
double I_ideal = U*1000/R; //Ideal constant current, mA

double deltaC = 0;
double deltaPhi = 0;
double slope = deltaC / (0.25 * 2 * M_PI * pow(10,6));
double slopePhi = deltaPhi / (0.25 * 2 * M_PI * pow(10,6));

double Ctrim(double w) {
  return slope*(w-wRes);
}

double C(double w) {
  return Cmain + Ctrim(w)*pow(10,-12);
}

double Cpf = C(wRes)*pow(10,12);

//500 kHz scan width centered on 213 MHz
double wLow = 2*M_PI*(213-0.25)*pow(10,6);
double wHigh = 2*M_PI*(213+0.25)*pow(10,6);



//----------CABLE CHARACTERISTICS-----------

double alpha = 0.0343;
double beta1 = 4.752 * pow(10,-9);
double trim = 7*0.5;

double beta(double w) {
  return beta1*w;
}

double cableImpedance = 50; //Cable Impedance ~ 50 ohm
double S = 2*cableImpedance*alpha; //ohm/meter

double D = 10.27*pow(10,-11); // F/m
double M = 2.542*pow(10,-7); // H/m

double complex Z0(double w) {
  double realPart = (M/D);
  double imPart = (-1*S)/(w*D);
  double complex squaredValue = realPart + I*imPart;
  return csqrt(squaredValue);
}

double complex Zres = Z0(wRes);

double vel = 1/(beta(1));

double lambda(double w) {
  return vel/f;
}

double l = trim*lambda(wRes); //Length of cable



FILE *butxii = fopen("data/PROTON.DAT","r");
FILE *butxi = fopen("data/DPROTON.DAT","r");
FILE *Vback = fopen("data/Backgmd.dat","r");
FILE *Vreal = fopen("data/Backreal.dat","r");

FILE *output = fopen("output.dat", "w");


fclose(butxii);
fclose(butxi);
fclose(Vback);
fclose(Vreal);
fclose(output);

int main() {
  printf("%lf\n",l);   
}
