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
    double QQ, t;
    double xi, M2, tau, e, tmin;
    double Gamma;
    double jcob;

    TLorentzVector *QP, *D, *P;

    double kkp;
    double *kd, *kpd, *kP, *kpP, *kqp, *kpqp, *dd, *Pq, *Pqp, *qd, *qpd;

    double kk_T, kkp_T;
    double *kqp_T, *kd_T, *dd_T, *kpqp_T, *kP_T, *kpP_T, *qpP_T, *kpd_T, *qpd_T;

    double *Dplus, *Dminus;

    double F1, F2;

    double *xbhUU;

    // OO languages should probably use a hashmap instead of an array for mapping phi values to indices
    double *phiValues;
    unsigned long lengthOfPhiValues;
} TVA1_UU;



void TVA1_UU_Init(TVA1_UU * const restrict self, const double QQ, const double x, const double t, const double k, const double F1, const double F2, double * const restrict phiValues, const unsigned long lengthOfPhiValues);
void TVA1_UU_Destruct(const TVA1_UU * const restrict self);
unsigned long TVA1_UU_getPhiIndex(const TVA1_UU * const restrict self, const double phi);
double TVA1_UU_TProduct(const TLorentzVector * const restrict v1, const TLorentzVector * const restrict v2);
double TVA1_UU_getBHUU_plus_getIUU(const TVA1_UU * const restrict self, const double phi, const double ReH, const double ReE, const double ReHtilde);



void TVA1_UU_Init(TVA1_UU * const restrict self, const double QQ, const double x, const double t, const double k, const double F1, const double F2, double * const restrict phiValues, const unsigned long lengthOfPhiValues) {
    self->lengthOfPhiValues = lengthOfPhiValues;

    self->F1 = F1;
    self->F2 = F2;

    self->QQ = QQ;
    self->t = t;
    self->M2 = M * M;
    self->jcob = 2 * PI;

    const double y = QQ / (2 * M * k * x);
    const double gg = 4 * self->M2 * x * x / QQ;
    self->e = (1 - y - (y * y * (gg / 4))) / (1 - y + (y * y / 2) + (y * y * (gg / 4)));
    self->xi = x * (1 + t / (2 * QQ)) / (2 - x + (x * t / QQ));
    self->tmin = -1 * QQ * (1 - sqrt(1 + gg) + (gg / 2)) / (x * (1 - sqrt(1 + gg) + (gg / (2 * x))));
    const double kpr = k * (1 - y);
    const double qp = ((t / 2) / M) + k - kpr;
    const double cth = -1 / sqrt(1 + gg) * (1 + ((gg / 2) * (1 + (t / QQ)) / (1 + (x * t / QQ))));
    const double theta = acos(cth);
    const double sthl = sqrt(gg) / sqrt(1 + gg) * (sqrt(1 - y - (y * y * gg / 4)));
    const double cthl = -1 / sqrt(1 + gg) * (1 + (y * gg / 2));
    self->tau = -0.25 * t / self->M2;

    TLorentzVector K, KP, Q, p;

    TLorentzVector_SetPxPyPzE(
        &K, 
        k * sthl, 
        0.0, 
        k * cthl, 
        k
    );
    TLorentzVector_SetPxPyPzE(
        &KP, 
        TLorentzVector_Parentheses_Operator(K, 0), 
        0.0, 
        k * (cthl + (y * sqrt(1 + gg))), 
        kpr
    );
    Q = TLorentzVector_Minus_Operator(K, KP);
    TLorentzVector_SetPxPyPzE(&p, 0.0, 0.0, 0.0, M);
    const double s = TLorentzVector_Multiplication_Operator(TLorentzVector_Plus_Operator(p, K), TLorentzVector_Plus_Operator(p, K));
    self->Gamma = 1 / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16 / (s - self->M2) / (s - self->M2) / sqrt(1 + gg) / x;

    self->kkp  = TLorentzVector_Multiplication_Operator(K, KP);

    self->kk_T   = TVA1_UU_TProduct(&K, &K);
    self->kkp_T  = self->kk_T;

    self->phiValues = malloc(sizeof(double) * lengthOfPhiValues);
    self->xbhUU = malloc(sizeof(double) * lengthOfPhiValues);

    self->QP =     malloc(sizeof(TLorentzVector) * lengthOfPhiValues);
    self->D =      malloc(sizeof(TLorentzVector) * lengthOfPhiValues);
    self->P =      malloc(sizeof(TLorentzVector) * lengthOfPhiValues);

    self->kd =     malloc(sizeof(double) * lengthOfPhiValues);
    self->kpd =    malloc(sizeof(double) * lengthOfPhiValues);
    self->kP =     malloc(sizeof(double) * lengthOfPhiValues);
    self->kpP =    malloc(sizeof(double) * lengthOfPhiValues);
    self->kqp =    malloc(sizeof(double) * lengthOfPhiValues);
    self->kpqp =   malloc(sizeof(double) * lengthOfPhiValues);
    self->dd =     malloc(sizeof(double) * lengthOfPhiValues);
    self->Pq =     malloc(sizeof(double) * lengthOfPhiValues);
    self->Pqp =    malloc(sizeof(double) * lengthOfPhiValues);
    self->qd =     malloc(sizeof(double) * lengthOfPhiValues);
    self->qpd =    malloc(sizeof(double) * lengthOfPhiValues);

    self->kqp_T =  malloc(sizeof(double) * lengthOfPhiValues);
    self->kd_T =   malloc(sizeof(double) * lengthOfPhiValues);
    self->dd_T =   malloc(sizeof(double) * lengthOfPhiValues);
    self->kpqp_T = malloc(sizeof(double) * lengthOfPhiValues);
    self->kP_T =   malloc(sizeof(double) * lengthOfPhiValues);
    self->kpP_T =  malloc(sizeof(double) * lengthOfPhiValues);
    self->qpP_T =  malloc(sizeof(double) * lengthOfPhiValues);
    self->kpd_T =  malloc(sizeof(double) * lengthOfPhiValues);
    self->qpd_T =  malloc(sizeof(double) * lengthOfPhiValues);

    self->Dplus =  malloc(sizeof(double) * lengthOfPhiValues);
    self->Dminus = malloc(sizeof(double) * lengthOfPhiValues);


    for (unsigned long i = 0; i < lengthOfPhiValues; i++) {
        self->phiValues[i] = phiValues[i];

        // Set4VectorsPhiDep
        TLorentzVector_SetPxPyPzE(
            &self->QP[i], 
            qp * sin(theta) * cos(phiValues[i] * RAD), 
            qp * sin(theta) * sin(phiValues[i] * RAD), 
            qp * cos(theta), 
            qp
        );
        self->D[i] = TLorentzVector_Minus_Operator(Q, self->QP[i]);
        TLorentzVector pp = TLorentzVector_Plus_Operator(p, self->D[i]);
        self->P[i] = TLorentzVector_Plus_Operator(p, pp);
        TLorentzVector_SetPxPyPzE(
            &self->P[i], 
            0.5 * TLorentzVector_Px(&self->P[i]),
            0.5 * TLorentzVector_Py(&self->P[i]),
            0.5 * TLorentzVector_Pz(&self->P[i]),
            0.5 * TLorentzVector_E(&self->P[i])
        );

        // Set4VectorProducts
        self->kd[i]   = TLorentzVector_Multiplication_Operator(K, self->D[i]);
        self->kpd[i]  = TLorentzVector_Multiplication_Operator(KP, self->D[i]);
        self->kP[i]   = TLorentzVector_Multiplication_Operator(K, self->P[i]);
        self->kpP[i]  = TLorentzVector_Multiplication_Operator(KP, self->P[i]);
        self->kqp[i]  = TLorentzVector_Multiplication_Operator(K, self->QP[i]);
        self->kpqp[i] = TLorentzVector_Multiplication_Operator(KP, self->QP[i]);
        self->dd[i]   = TLorentzVector_Multiplication_Operator(self->D[i], self->D[i]);
        self->Pq[i]   = TLorentzVector_Multiplication_Operator(self->P[i], Q);
        self->Pqp[i]  = TLorentzVector_Multiplication_Operator(self->P[i], self->QP[i]);
        self->qd[i]   = TLorentzVector_Multiplication_Operator(Q, self->D[i]);
        self->qpd[i]  = TLorentzVector_Multiplication_Operator(self->QP[i], self->D[i]);

        self->kqp_T[i]  = TVA1_UU_TProduct(&K, &self->QP[i]);
        self->kd_T[i]   = -1 * self->kqp_T[i];
        self->dd_T[i]   = TVA1_UU_TProduct(&self->D[i], &self->D[i]);
        self->kpqp_T[i] = self->kqp_T[i];
        self->kP_T[i]   = TVA1_UU_TProduct(&K, &self->P[i]);
        self->kpP_T[i]  = TVA1_UU_TProduct(&KP, &self->P[i]);
        self->qpP_T[i]  = TVA1_UU_TProduct(&self->QP[i], &self->P[i]);
        self->kpd_T[i]  = -1 * self->kqp_T[i];
        self->qpd_T[i]  = -1 * self->dd_T[i];

        self->Dplus[i]   = 0.5 / self->kpqp[i] - 0.5 / self->kqp[i];
        self->Dminus[i]  = -0.5 / self->kpqp[i] - 0.5 / self->kqp[i];


        // getBHUU
        const double AUUBH = (8. * self->M2) / (self->t * self->kqp[i] * self->kpqp[i]) * ( (4. * self->tau * (self->kP[i] * self->kP[i] + self->kpP[i] * self->kpP[i]) ) - ( (self->tau + 1.) * (self->kd[i] * self->kd[i] + self->kpd[i] * self->kpd[i]) ) );
        const double BUUBH = (16. * self->M2) / (self->t * self->kqp[i] * self->kpqp[i]) * (self->kd[i] * self->kd[i] + self->kpd[i] * self->kpd[i]);

        const double con_AUUBH = AUUBH * GeV2nb * self->jcob;
        const double con_BUUBH = BUUBH * GeV2nb * self->jcob;

        const double bhAUU = (self->Gamma/self->t) * con_AUUBH * ( F1 * F1 + self->tau * F2 * F2 );
        const double bhBUU = (self->Gamma/self->t) * con_BUUBH * ( self->tau * ( F1 + F2 ) * ( F1 + F2 ) );

        self->xbhUU[i] = bhAUU + bhBUU;
    }
}



unsigned long TVA1_UU_getPhiIndex(const TVA1_UU * const restrict self, const double phi) {
    for (unsigned long i = 0; i < self->lengthOfPhiValues; i++) {
        if (fabs(self->phiValues[i] - phi) < 0.000001) {
            return i;
        }
    }

    perror("Error: phi value not found\n");
    exit(1);
}



double TVA1_UU_TProduct(const TLorentzVector * const restrict v1, const TLorentzVector * const restrict v2) {
    return (TLorentzVector_Px(v1) * TLorentzVector_Px(v2)) + (TLorentzVector_Py(v1) * TLorentzVector_Py(v2));
}



double TVA1_UU_getBHUU_plus_getIUU(const TVA1_UU * const restrict self, const double phi, const double ReH, const double ReE, const double ReHtilde) {
    const unsigned long phiIndex = TVA1_UU_getPhiIndex(self, phi);

    // getIUU
    const double AUUI = -4. * cos( phi * RAD ) * ( self->Dplus[phiIndex] * ( ( self->kqp_T[phiIndex] - 2. * self->kk_T - 2. * self->kqp[phiIndex] ) * self->kpP[phiIndex] + ( 2. * self->kpqp[phiIndex] - 2. * self->kkp_T - self->kpqp_T[phiIndex] ) * self->kP[phiIndex] + self->kpqp[phiIndex] * self->kP_T[phiIndex] + self->kqp[phiIndex] * self->kpP_T[phiIndex] - 2.*self->kkp * self->kP_T[phiIndex] ) -
                        self->Dminus[phiIndex] * ( ( 2. * self->kkp - self->kpqp_T[phiIndex] - self->kkp_T ) * self->Pqp[phiIndex] + 2. * self->kkp * self->qpP_T[phiIndex] - self->kpqp[phiIndex] * self->kP_T[phiIndex] - self->kqp[phiIndex] * self->kpP_T[phiIndex] ) ) ;
    const double BUUI = -2. * self->xi * cos( phi * RAD ) * ( self->Dplus[phiIndex] * ( ( self->kqp_T[phiIndex] - 2. * self->kk_T - 2. * self->kqp[phiIndex] ) * self->kpd[phiIndex] + ( 2. * self->kpqp[phiIndex] - 2. * self->kkp_T - self->kpqp_T[phiIndex] ) * self->kd[phiIndex] + self->kpqp[phiIndex] * self->kd_T[phiIndex] + self->kqp[phiIndex] * self->kpd_T[phiIndex]- 2.*self->kkp * self->kd_T[phiIndex] ) -
                        self->Dminus[phiIndex] * ( ( 2. * self->kkp - self->kpqp_T[phiIndex] - self->kkp_T ) * self->qpd[phiIndex] + 2. * self->kkp * self->qpd_T[phiIndex] - self->kpqp[phiIndex] * self->kd_T[phiIndex] - self->kqp[phiIndex] * self->kpd_T[phiIndex] ) );
    const double CUUI = -2. * cos( phi * RAD ) * ( self->Dplus[phiIndex] * ( 2. * self->kkp * self->kd_T[phiIndex] - self->kpqp[phiIndex] * self->kd_T[phiIndex] - self->kqp[phiIndex] * self->kpd_T[phiIndex] + 4. * self->xi * self->kkp * self->kP_T[phiIndex] - 2. * self->xi * self->kpqp[phiIndex] * self->kP_T[phiIndex] - 2. * self->xi * self->kqp[phiIndex] * self->kpP_T[phiIndex] ) -
                        self->Dminus[phiIndex] * ( self->kkp * self->qpd_T[phiIndex] - self->kpqp[phiIndex] * self->kd_T[phiIndex] - self->kqp[phiIndex] * self->kpd_T[phiIndex] + 2. * self->xi * self->kkp * self->qpP_T[phiIndex] - 2. * self->xi * self->kpqp[phiIndex] * self->kP_T[phiIndex] - 2. * self->xi * self->kqp[phiIndex] * self->kpP_T[phiIndex]) );

    const double con_AUUI = AUUI * GeV2nb * self->jcob;
    const double con_BUUI = BUUI * GeV2nb * self->jcob;
    const double con_CUUI = CUUI * GeV2nb * self->jcob;

    const double iAUU = (self->Gamma/(fabs(self->t) * self->QQ)) * con_AUUI * ( self->F1 * ReH + self->tau * self->F2 * ReE );
    const double iBUU = (self->Gamma/(fabs(self->t) * self->QQ)) * con_BUUI * ( self->F1 + self->F2 ) * ( ReH + ReE );
    const double iCUU = (self->Gamma/(fabs(self->t) * self->QQ)) * con_CUUI * ( self->F1 + self->F2 ) * ReHtilde;

    const double xIUU = iAUU + iBUU + iCUU;


    // getBHUU + getIUU = xbhUU + (-1 * xIUU)
    return self->xbhUU[phiIndex] - xIUU;
}



void TVA1_UU_Destruct(const TVA1_UU * const restrict self) {
    free(self->phiValues);
    free(self->xbhUU);

    free(self->QP);
    free(self->D);
    free(self->P);

    free(self->kd);
    free(self->kpd);
    free(self->kP);
    free(self->kpP);
    free(self->kqp);
    free(self->kpqp);
    free(self->dd);
    free(self->Pq);
    free(self->Pqp);
    free(self->qd);
    free(self->qpd);

    free(self->kqp_T);
    free(self->kd_T);
    free(self->dd_T);
    free(self->kpqp_T);
    free(self->kP_T);
    free(self->kpP_T);
    free(self->qpP_T);
    free(self->kpd_T);
    free(self->qpd_T);

    free(self->Dplus);
    free(self->Dminus);
}

#endif // TVA1_UU_H_
