#ifndef BHDVCS_H_
#define BHDVCS_H_

#include <math.h>

#include "LorentzVector.h"

typedef struct {
    double ALP_INV; //  1 / Electromagnetic Fine Structure Constant
    double PI;
    double RAD;
    double M; // Mass of the proton in GeV
    double GeV2nb; //  Conversion from GeV to NanoBarn
    //  Elastic FF
    double F1;  //  Dirac FF - helicity conserving (non spin flip)
    double F2;  //  Pauli FF - helicity non-conserving (spin flip)

    double QQ, x, t, k; //  creo q no hace falta
    double y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M2, tau;

    //  4-momentum vectors
    LorentzVector K;
    LorentzVector KP;
    LorentzVector Q;
    LorentzVector QP;
    LorentzVector D;
    LorentzVector p;
    LorentzVector P;

    //  4 - vector products independent of phi
    double kkp;
    double kq;
    double kp;
    double kpp;

    //  4 - vector products dependent of phi
    double kd, kpd, kP, kpP, kqp, kpqp, dd, Pq, Pqp, qd, qpd;

    LorentzVector pp;

    //      // KK_T, KQP_T, KKP_T, KXQP_T, KD_T, DD_T
    double  kk_T, kqp_T, kkp_T, kd_T, dd_T;

    double kpqp_T, kP_T, kpP_T, qpP_T, kpd_T, qpd_T;

    double s;     //  Mandelstam variable s which is the center of mass energy
    double Gamma; //  Factor in front of the cross section
    double jcob;  // Defurne's Jacobian

    double  AUUBH, BUUBH; //  Coefficients of the BH unpolarized structure function FUU_BH
    double  AUUI, BUUI, CUUI; //  Coefficients of the BHDVCS interference unpolarized structure function FUU_I
    double  con_AUUBH, con_BUUBH, con_AUUI, con_BUUI, con_CUUI;  //  Coefficients times the conversion to nb and the jacobian
    double  bhAUU, bhBUU; //  Auu and Buu term of the BH cross section
    double  iAUU, iBUU, iCUU; //  Terms of the interference containing AUUI, BUUI and CUUI
    double  xbhUU; //  Unpolarized BH cross section
    double  xIUU; //  Unpolarized interference cross section
} BHDVCS;

void BHDVCS_Init(BHDVCS * restrict self);
double BHDVCS_TProduct(LorentzVector * restrict v1, LorentzVector * restrict v2);
void BHDVCS_SetKinematics(BHDVCS * restrict self, double _QQ, double _x, double _t, double _k);
void BHDVCS_Set4VectorsPhiDep(BHDVCS * restrict self, int phi);
void BHDVCS_Set4VectorProducts(BHDVCS * restrict self, int phi);
double BHDVCS_GetBHUUxs(BHDVCS * restrict self, int phi, double F1, double F2);
double BHDVCS_GetIUUxs(BHDVCS * restrict self, int phi, double F1, double F2, double ReH, double ReE, double ReHtilde);
double BHDVCS_TotalUUXS(BHDVCS * restrict self, int phi, double k, double QQ, double xb, double t, 
                        double F1, double F2, double dvcs, double ReH, double ReE, double ReHtilde);




void BHDVCS_Init(BHDVCS * restrict self) {
    self->ALP_INV = 137.0359998;
    self->PI = 3.1415926535;
    self->RAD = self->PI / 180.;
    self->M = 0.938272;
    self->GeV2nb = .389379 * 1000000;
}


double BHDVCS_TProduct(LorentzVector * restrict v1, LorentzVector * restrict v2) {
    return (v1->x * v2->x) + (v1->y * v2->y);
}

void BHDVCS_SetKinematics(BHDVCS * restrict self, double _QQ, double _x, double _t, double _k) {
    self->QQ = _QQ; //Q^2 value
    self->x = _x;   //Bjorken x
    self->t = _t;   //momentum transfer squared
    self->k = _k;   //Electron Beam Energy
    self->M2 = self->M*self->M; //Mass of the proton  squared in GeV
    //fractional energy of virtual photon
    self->y = self->QQ / ( 2. * self->M * self->k * self->x ); // From eq. (23) where gamma is substituted from eq (12c)
    //squared gamma variable ratio of virtuality to energy of virtual photon
    self->gg = 4. * self->M2 * self->x * self->x / self->QQ; // This is gamma^2 [from eq. (12c)]
    //ratio of longitudinal to transverse virtual photon flux
    self->e = ( 1 - self->y - ( self->y * self->y * (self->gg / 4.) ) ) / ( 1. - self->y + (self->y * self->y / 2.) + ( self->y * self->y * (self->gg / 4.) ) ); // epsilon eq. (32)
    //Skewness parameter
    self->xi = 1. * self->x * ( ( 1. + self->t / ( 2. * self->QQ ) ) / ( 2. - self->x + self->x * self->t / self->QQ ) ); // skewness parameter eq. (12b) dnote: there is a minus sign on the write up that shouldn't be there
    //Minimum t value
    self->tmin = ( self->QQ * ( 1. - sqrt( 1. + self->gg ) + self->gg / 2. ) ) / ( self->x * ( 1. - sqrt( 1. + self->gg ) + self->gg / ( 2.* self->x ) ) ); // minimum t eq. (29)
    //Final Lepton energy
    self->kpr = self->k * ( 1. - self->y ); // k' from eq. (23)
    //outgoing photon energy
    self->qp = self->t / 2. / self->M + self->k - self->kpr; //q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
    //Final proton Energy
    self->po = self->M - self->t / 2. / self->M; // This is p'_0 from eq. (28b)
    self->pmag = sqrt( ( -1* self->t ) * ( 1. - (self->t / (4. * self->M * self->M ))) ); // p' magnitude from eq. (28b)
    //Angular Kinematics of outgoing photon
    self->cth = -1. / sqrt( 1. + self->gg ) * ( 1. + self->gg / 2. * ( 1. + self->t / self->QQ ) / ( 1. + self->x * self->t / self->QQ ) ); // This is cos(theta) eq. (26)
    self->theta = acos(self->cth); // theta angle
    //print('Theta: ', self->theta)
    //Lepton Angle Kinematics of initial lepton
    self->sthl = sqrt( self->gg ) / sqrt( 1. + self->gg ) * ( sqrt ( 1. - self->y - self->y * self->y * self->gg / 4. ) ); // sin(theta_l) from eq. (22a)
    self->cthl = -1. / sqrt( 1. + self->gg ) * ( 1. + self->y * self->gg / 2. );  // cos(theta_l) from eq. (22a)
    //ratio of momentum transfer to proton mass
    self->tau = -0.25 * self->t / self->M2;

    // phi independent 4 - momenta vectors defined on eq. (21) -------------
    self->K = LorentzVector_SetPxPyPzE(self->k * self->sthl, 0.0, self->k * self->cthl, self->k);
    self->KP = LorentzVector_SetPxPyPzE(self->K.x, 0.0, self->k * ( self->cthl + self->y * sqrt( 1. + self->gg ) ), self->kpr);
    self->Q = LorentzVector_sub(self->K, self->KP);
    self->p = LorentzVector_SetPxPyPzE(0.0, 0.0, 0.0, self->M);

    // Sets the Mandelstam variable s which is the center of mass energy
    self->s = LorentzVector_mul(LorentzVector_add(self->p, self->K), LorentzVector_add(self->p, self->K));

    // The Gamma factor in front of the cross section
    self->Gamma = 1. / self->ALP_INV / self->ALP_INV / self->ALP_INV / self->PI / self->PI / 16. / ( self->s - self->M2 ) / ( self->s - self->M2 ) / sqrt( 1. + self->gg ) / self->x;

    // Defurne's Jacobian
    self->jcob = 1./ ( 2. * self->M * self->x * self->K.t ) * 2. * self->PI * 2.;
    //print("Jacobian: ", self->jcob)
    //___________________________________________________________________________________

}

void BHDVCS_Set4VectorsPhiDep(BHDVCS * restrict self, int phi) {

        // phi dependent 4 - momenta vectors defined on eq. (21) -------------

        self->QP = LorentzVector_SetPxPyPzE(self->qp * sin(self->theta) * cos( phi * self->RAD ), self->qp * sin(self->theta) * sin( phi * self->RAD ), self->qp * cos(self->theta), self->qp);
        self->D = LorentzVector_sub(self->Q, self->QP); // delta vector eq. (12a)
        //print(self->D, "\n", self->Q, "\n", self->QP)
        self->pp = LorentzVector_add(self->p, self->D); // p' from eq. (21)
        self->P = LorentzVector_add(self->p, self->pp);
        self->P = LorentzVector_SetPxPyPzE(.5 * self->P.x, .5 * self->P.y, .5 * self->P.z, .5 * self->P.t);

        //____________________________________________________________________________________
}


void BHDVCS_Set4VectorProducts(BHDVCS * restrict self, int phi) {
        // phi is unused in this function
        phi += 3;

        // 4-vectors products (phi - independent)
        self->kkp  = LorentzVector_mul(self->K, self->KP);   //(kk')
        self->kq   = LorentzVector_mul(self->K, self->Q);    //(kq)
        self->kp   = LorentzVector_mul(self->K, self->p);    //(pk)
        self->kpp  = LorentzVector_mul(self->KP, self->p);   //(pk')

        // 4-vectors products (phi - dependent)
        self->kd   = LorentzVector_mul(self->K, self->D);    //(kd)
        self->kpd  = LorentzVector_mul(self->KP, self->D);   //(k'd)
        self->kP   = LorentzVector_mul(self->K, self->P);    //(kP)
        self->kpP  = LorentzVector_mul(self->KP, self->P);   //(k'P)
        self->kqp  = LorentzVector_mul(self->K, self->QP);   //(kq')
        self->kpqp = LorentzVector_mul(self->KP, self->QP);  //(k'q')
        self->dd   = LorentzVector_mul(self->D, self->D);    //(dd)
        self->Pq   = LorentzVector_mul(self->P, self->Q);    //(Pq)
        self->Pqp  = LorentzVector_mul(self->P, self->QP);   //(Pq')
        self->qd   = LorentzVector_mul(self->Q, self->D);    //(qd)
        self->qpd  = LorentzVector_mul(self->QP, self->D);   //(q'd)

        // //Transverse vector products defined after eq.(241c) -----------------------
        self->kk_T   = BHDVCS_TProduct(&self->K, &self->K);
        self->kkp_T  = self->kk_T;
        self->kqp_T  = BHDVCS_TProduct(&self->K, &self->QP);
        self->kd_T   = -1.* self->kqp_T;
        self->dd_T   = BHDVCS_TProduct(&self->D, &self->D);
        self->kpqp_T = BHDVCS_TProduct(&self->KP, &self->QP);
        self->kP_T   = BHDVCS_TProduct(&self->K, &self->P);
        self->kpP_T  = BHDVCS_TProduct(&self->KP, &self->P);
        self->qpP_T  = BHDVCS_TProduct(&self->QP, &self->P);
        self->kpd_T  = BHDVCS_TProduct(&self->KP, &self->D);
        self->qpd_T  = BHDVCS_TProduct(&self->QP, &self->D);

        //____________________________________________________________________________________
}

double BHDVCS_GetBHUUxs(BHDVCS * restrict self, int phi, double F1, double F2) {

    BHDVCS_Set4VectorsPhiDep(self, phi);
    BHDVCS_Set4VectorProducts(self, phi);

    // Coefficients of the BH unpolarized structure function FUUBH
    self->AUUBH = ( (8. * self->M2) / (self->t * self->kqp * self->kpqp) ) * ( (4. * self->tau * (self->kP * self->kP + self->kpP * self->kpP) ) - ( (self->tau + 1.) * (self->kd * self->kd + self->kpd * self->kpd) ) );
    self->BUUBH = ( (16. * self->M2) / (self->t* self->kqp * self->kpqp) ) * (self->kd * self->kd + self->kpd * self->kpd);

    // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
    // I multiply by 2 because I think Auu and Buu are missing a factor 2
    self->con_AUUBH = 2. * self->AUUBH * self->GeV2nb * self->jcob;
    self->con_BUUBH = 2. * self->BUUBH * self->GeV2nb * self->jcob;

    // Unpolarized Coefficients multiplied by the Form Factors
    self->bhAUU = (self->Gamma/self->t) * self->con_AUUBH * ( F1 * F1 + self->tau * F2 * F2 );
    self->bhBUU = (self->Gamma/self->t) * self->con_BUUBH * ( self->tau * ( F1 + F2 ) * ( F1 + F2 ) );

    // Unpolarized BH cross section
    self->xbhUU = self->bhAUU + self->bhBUU;

    return self->xbhUU;

    //____________________________________________________________________________________

}

double BHDVCS_GetIUUxs(BHDVCS * restrict self, int phi, double F1, double F2, double ReH, double ReE, double ReHtilde) {
    BHDVCS_Set4VectorsPhiDep(self, phi);
    BHDVCS_Set4VectorProducts(self, phi);

    // Interference coefficients given on eq. (241a,b,c)--------------------
    self->AUUI = -4.0 / (self->kqp * self->kpqp) * (( self->QQ + self->t ) * ( 2.0 * ( self->kP + self->kpP ) * self->kk_T + ( self->Pq * self->kqp_T ) + 2.* ( self->kpP * self->kqp ) - 2.* ( self->kP * self->kpqp ) + self->kpqp * self->kP_T + self->kqp * self->kpP_T - 2. * self->kkp * self->kP_T )
        + ( self->QQ - self->t + 4.* self->kd ) * ( self->Pqp * ( self->kkp_T + self->kqp_T - 2.* self->kkp ) + 2.* self->kkp * self->qpP_T - self->kpqp * self->kP_T - self->kqp * self->kpP_T ) );

    self->BUUI = 2.0 * self->xi / ( self->kqp * self->kpqp) * ( ( self->QQ + self->t ) * ( 2.* self->kk_T * ( self->kd + self->kpd ) + self->kqp_T * ( self->qd - self->kqp - self->kpqp + 2. * self->kkp ) + 2.* self->kqp * self->kpd - 2.* self->kpqp * self->kd ) +
                                                ( self->QQ - self->t + 4.* self->kd ) * ( ( self->kk_T - 2.* self->kkp ) * self->qpd - self->kkp * self->dd_T - 2.* self->kd_T * self->kqp ) );
    self->CUUI = 2.0 / ( self->kqp * self->kpqp) * ( -1. * ( self->QQ + self->t ) * ( 2.* self->kkp - self->kpqp - self->kqp + 2.* self->xi * (2.* self->kkp * self->kP_T - self->kpqp * self->kP_T - self->kqp * self->kpP_T) ) * self->kd_T +
                                                ( self->QQ - self->t + 4.* self->kd ) * ( ( self->kqp + self->kpqp ) * self->kd_T + self->dd_T * self->kkp + 2.* self->xi * ( self->kkp * self->qpP_T - self->kpqp * self->kP_T - self->kqp * self->kpP_T ) ) );

    // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian

    //print(self->AUUI, self->GeV2nb, self->jcob)
    self->con_AUUI = self->AUUI * self->GeV2nb * self->jcob;
    self->con_BUUI = self->BUUI * self->GeV2nb * self->jcob;
    self->con_CUUI = self->CUUI * self->GeV2nb * self->jcob;

    //Unpolarized Coefficients multiplied by the Form Factors

    self->iAUU = (self->Gamma/(-self->t * self->QQ)) * cos( phi * self->RAD ) * self->con_AUUI * ( F1 * ReH + self->tau * F2 * ReE );
    self->iBUU = (self->Gamma/(-self->t * self->QQ)) * cos( phi * self->RAD ) * self->con_BUUI * ( F1 + F2 ) * ( ReH + ReE );
    self->iCUU = (self->Gamma/(-self->t * self->QQ)) * cos( phi * self->RAD ) * self->con_CUUI * ( F1 + F2 ) * ReHtilde;


    // Unpolarized BH-DVCS interference cross section
    self->xIUU = self->iAUU + self->iBUU + self->iCUU;

    return self->xIUU;
}

double BHDVCS_TotalUUXS(BHDVCS * restrict self, int phi, double k, double QQ, double xb, double t, 
                        double F1, double F2, double dvcs, double ReH, double ReE, double ReHtilde) {
    // originally named TotalUUXS_curve_fit, but original TotalUUXS is unnecessary

    // Set QQ, xB, t and k and calculate 4-vector products
    BHDVCS_SetKinematics(self, QQ, xb, t, k);
    BHDVCS_Set4VectorsPhiDep(self, phi);
    BHDVCS_Set4VectorProducts(self, phi);

    double xsbhuu	 = BHDVCS_GetBHUUxs(self, phi, F1, F2);
    double xsiuu	 = BHDVCS_GetIUUxs(self, phi, F1, F2, ReH, ReE, ReHtilde);


    double tot_sigma_uu = xsbhuu + xsiuu +  dvcs; // Constant added to account for DVCS contribution

    return tot_sigma_uu;
}

#endif // BHDVCS_H_
