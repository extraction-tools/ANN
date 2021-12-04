#ifndef TVA1_UU_H_
#define TVA1_UU_H_



#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "TLorentzVector.h"



#define ALP_INV 137.0359998
#define PI 3.1415926535
#define RAD (PI / 180.)
#define M 0.938272
#define GeV2nb 389379



typedef struct {
    double QQ, t, Gamma, tau, F1, F2,        // kinematic variables
           *con_AUUI, *con_BUUI, *con_CUUI,  // intermediate getIUU computations
           *xbhUU,                           // getBHUU results
           *phiValues;                       // values of phi
    unsigned long numPhiValues;
} TVA1_UU;



void TVA1_UU_Init(TVA1_UU * const __restrict__ self, const double QQ, const double x, const double t, const double k, const double F1, const double F2, const double * const __restrict__ phiValues, const unsigned long numPhiValues);
void TVA1_UU_Destruct(const TVA1_UU * const __restrict__ self);
unsigned long TVA1_UU_getPhiIndex(const TVA1_UU * const __restrict__ self, const double phi);
double TVA1_UU_TProduct(const TLorentzVector * const __restrict__ v1, const TLorentzVector * const __restrict__ v2);
double TVA1_UU_getBHUU_plus_getIUU(const TVA1_UU * const __restrict__ self, const double phi, const double ReH, const double ReE, const double ReHtilde);



void TVA1_UU_Init(TVA1_UU * const __restrict__ self, const double QQ, const double x, const double t, const double k, const double F1, const double F2, const double * const __restrict__ phiValues, const unsigned long numPhiValues) {
    const double M2 = M * M;
    const double jcob = 2 * PI;
    const double y = QQ / (2 * M * k * x);
    const double gg = 4 * M2 * x * x / QQ;
    const double xi = x * (1 + t / (2 * QQ)) / (2 - x + (x * t / QQ));
    const double kpr = k * (1 - y);
    const double qp = ((t / 2) / M) + k - kpr;
    const double cth = -1 / sqrt(1 + gg) * (1 + ((gg / 2) * (1 + (t / QQ)) / (1 + (x * t / QQ))));
    const double theta = acos(cth);
    const double sthl = sqrt(gg) / sqrt(1 + gg) * (sqrt(1 - y - (y * y * gg / 4)));
    const double cthl = -1 / sqrt(1 + gg) * (1 + (y * gg / 2));

    const TLorentzVector K = TLorentzVector_Init(k * sthl, 0.0, k * cthl, k);
    const TLorentzVector KP = TLorentzVector_Init(TLorentzVector_paren_op(K, 0), 0.0, k * (cthl + (y * sqrt(1 + gg))), kpr);
    const TLorentzVector Q = TLorentzVector_minus(K, KP);
    const TLorentzVector p = TLorentzVector_Init(0.0, 0.0, 0.0, M);
    
    const double s = TLorentzVector_mul(TLorentzVector_plus(p, K), TLorentzVector_plus(p, K));
    const double kkp = TLorentzVector_mul(K, KP);
    const double kk_T = TVA1_UU_TProduct(&K, &K);
    const double kkp_T = kk_T;

    self->numPhiValues = numPhiValues;
    self->F1 = F1;
    self->F2 = F2;
    self->QQ = QQ;
    self->t = t;
    self->tau = -0.25 * t / M2;
    self->Gamma = 1 / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16 / (s - M2) / (s - M2) / sqrt(1 + gg) / x;

    self->phiValues = (double *) malloc(sizeof(double) * numPhiValues);
    self->con_AUUI =  (double *) malloc(sizeof(double) * numPhiValues);
    self->con_BUUI =  (double *) malloc(sizeof(double) * numPhiValues);
    self->con_CUUI =  (double *) malloc(sizeof(double) * numPhiValues);
    self->xbhUU =     (double *) malloc(sizeof(double) * numPhiValues);

    for (unsigned long i = 0; i < numPhiValues; i++) {
        // Set4VectorsPhiDep
        const TLorentzVector QP = TLorentzVector_Init(qp * sin(theta) * cos(phiValues[i] * RAD), qp * sin(theta) * sin(phiValues[i] * RAD), qp * cos(theta), qp);
        const TLorentzVector D = TLorentzVector_minus(Q, QP);
        const TLorentzVector pp = TLorentzVector_plus(p, D);
        const TLorentzVector P = TLorentzVector_Init((p.x + pp.x) / 2, (p.y + pp.y) / 2, (p.z + pp.z) / 2, (p.t + pp.t) / 2);


        // Set4VectorProducts
        const double kd   = TLorentzVector_mul(K, D);
        const double kpd  = TLorentzVector_mul(KP, D);
        const double kP   = TLorentzVector_mul(K, P);
        const double kpP  = TLorentzVector_mul(KP, P);
        const double kqp  = TLorentzVector_mul(K, QP);
        const double kpqp = TLorentzVector_mul(KP, QP);
        const double Pqp  = TLorentzVector_mul(P, QP);
        const double qpd  = TLorentzVector_mul(QP, D);

        const double kqp_T  = TVA1_UU_TProduct(&K, &QP);
        const double kd_T   = -1 * kqp_T;
        const double dd_T   = TVA1_UU_TProduct(&D, &D);
        const double kpqp_T = kqp_T;
        const double kP_T   = TVA1_UU_TProduct(&K, &P);
        const double kpP_T  = TVA1_UU_TProduct(&KP, &P);
        const double qpP_T  = TVA1_UU_TProduct(&QP, &P);
        const double kpd_T  = -1 * kqp_T;
        const double qpd_T  = -1 * dd_T;

        const double Dplus  = (0.5 / kpqp) - (0.5 / kqp);
        const double Dminus = (-0.5 / kpqp) - (0.5 / kqp);


        // getBHUU
        const double AUUBH = (8. * M2) / (self->t * kqp * kpqp) * ( (4. * self->tau * (kP * kP + kpP * kpP) ) - ( (self->tau + 1.) * (kd * kd + kpd * kpd) ) );
        const double BUUBH = (16. * M2) / (self->t * kqp * kpqp) * (kd * kd + kpd * kpd);

        const double con_AUUBH = AUUBH * GeV2nb * jcob;
        const double con_BUUBH = BUUBH * GeV2nb * jcob;

        const double bhAUU = (self->Gamma/self->t) * con_AUUBH * ( F1 * F1 + self->tau * F2 * F2 );
        const double bhBUU = (self->Gamma/self->t) * con_BUUBH * ( self->tau * ( F1 + F2 ) * ( F1 + F2 ) );


        // first half of getIUU
        const double AUUI = -4. * cos( phiValues[i] * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpP + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kP + kpqp * kP_T + kqp * kpP_T - 2.*kkp * kP_T ) - Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * Pqp + 2. * kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) ;
        const double BUUI = -2. * xi * cos( phiValues[i] * RAD ) * ( Dplus * ( ( kqp_T - 2. * kk_T - 2. * kqp ) * kpd + ( 2. * kpqp - 2. * kkp_T - kpqp_T ) * kd + kpqp * kd_T + kqp * kpd_T- 2.*kkp * kd_T ) - Dminus * ( ( 2. * kkp - kpqp_T - kkp_T ) * qpd + 2. * kkp * qpd_T - kpqp * kd_T - kqp * kpd_T ) );
        const double CUUI = -2. * cos( phiValues[i] * RAD ) * ( Dplus * ( 2. * kkp * kd_T - kpqp * kd_T - kqp * kpd_T + 4. * xi * kkp * kP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T ) - Dminus * ( kkp * qpd_T - kpqp * kd_T - kqp * kpd_T + 2. * xi * kkp * qpP_T - 2. * xi * kpqp * kP_T - 2. * xi * kqp * kpP_T) );


        self->phiValues[i] = phiValues[i];
        self->xbhUU[i] = bhAUU + bhBUU;
        self->con_AUUI[i] = AUUI * GeV2nb * jcob;
        self->con_BUUI[i] = BUUI * GeV2nb * jcob;
        self->con_CUUI[i] = CUUI * GeV2nb * jcob;
    }
}



unsigned long TVA1_UU_getPhiIndex(const TVA1_UU * const __restrict__ self, const double phi) {
    for (unsigned long i = 0; i < self->numPhiValues; i++) {
        if (fabs(self->phiValues[i] - phi) < 0.000001) {
            return i;
        }
    }

    perror("Error: phi value not found\n");
    exit(255);
}



double TVA1_UU_TProduct(const TLorentzVector * const __restrict__ v1, const TLorentzVector * const __restrict__ v2) {
    return (TLorentzVector_Px(v1) * TLorentzVector_Px(v2)) + (TLorentzVector_Py(v1) * TLorentzVector_Py(v2));
}



double TVA1_UU_getBHUU_plus_getIUU(const TVA1_UU * const __restrict__ self, const double phi, const double ReH, const double ReE, const double ReHtilde) {
    const unsigned long phiIndex = TVA1_UU_getPhiIndex(self, phi);

    // second half of getIUU
    const double something = (self->Gamma/(fabs(self->t) * self->QQ));

    const double iAUU = something * self->con_AUUI[phiIndex] * ((self->F1 * ReH) + (self->tau * self->F2 * ReE));
    const double iBUU = something * self->con_BUUI[phiIndex] * (self->F1 + self->F2) * (ReH + ReE);
    const double iCUU = something * self->con_CUUI[phiIndex] * (self->F1 + self->F2) * ReHtilde;

    const double xIUU = iAUU + iBUU + iCUU;

    // getBHUU + getIUU = xbhUU + (-1 * xIUU)
    return self->xbhUU[phiIndex] - xIUU;
}



void TVA1_UU_Destruct(const TVA1_UU * const __restrict__ self) {
    free(self->phiValues);
    free(self->con_AUUI);
    free(self->con_BUUI);
    free(self->con_CUUI);
    free(self->xbhUU);
}



#endif // TVA1_UU_H_
