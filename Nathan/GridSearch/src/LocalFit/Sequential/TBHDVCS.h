#ifndef BHDVCS_H_
#define BHDVCS_H_

#include <math.h>

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
} BHDVCS;

void BHDVCS_Init(BHDVCS * __restrict__ self, const double QQ, const double x, const double t, const double k, const double F1, const double F2, const double * const __restrict__ phiValues, const unsigned long numPhiValues);
double BHDVCS_TProduct(const TLorentzVector * __restrict__ v1, const TLorentzVector * __restrict__ v2);
double BHDVCS_GetIUUxs(const BHDVCS * __restrict__ self, double phi, double ReH, double ReE, double ReHtilde);
double BHDVCS_getBHUU_plus_getIUU(const BHDVCS * const __restrict__ self, const double phi, const double ReH, const double ReE, const double ReHtilde);
void BHDVCS_Destruct(const BHDVCS * const __restrict__ self);
unsigned long BHDVCS_getPhiIndex(const BHDVCS * const __restrict__ self, const double phi);

unsigned long BHDVCS_getPhiIndex(const BHDVCS * const __restrict__ self, const double phi) {
    for (unsigned long i = 0; i < self->numPhiValues; i++) {
        if (fabs(self->phiValues[i] - phi) < 0.000001) {
            return i;
        }
    }

    perror("Error: phi value not found\n");
    exit(255);
}



void BHDVCS_Init(BHDVCS * __restrict__ self, const double QQ, const double x, const double t, const double k, const double F1, const double F2, const double * const __restrict__ phiValues, const unsigned long numPhiValues) {
    self->t  = t;
    self->QQ = QQ;
    self->F1 = F1;
    self->F2 = F2;
    self->numPhiValues = numPhiValues;

    // Set Kinematics
    const double M2 = M*M; //Mass of the proton  squared in GeV
    //fractional energy of virtual photon
    const double y = QQ / ( 2. * M * k * x ); // From eq. (23) where gamma is substituted from eq (12c)
    //squared gamma variable ratio of virtuality to energy of virtual photon
    const double gg = 4. * M2 * x * x / QQ; // This is gamma^2 [from eq. (12c)]
    //ratio of longitudinal to transverse virtual photon flux
    //const double e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); // epsilon eq. (32)
    //Skewness parameter
    const double xi = 1. * x * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - x + x * t / QQ ) ); // skewness parameter eq. (12b) dnote: there is a minus sign on the write up that shouldn't be there
    //Minimum t value
    //const double tmin = ( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( x * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* x ) ) ); // minimum t eq. (29)
    //Final Lepton energy
    const double kpr = k * ( 1. - y ); // k' from eq. (23)
    //outgoing photon energy
    const double qp = t / 2. / M + k - kpr; //q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
    //Final proton Energy
    //const double po = M - t / 2. / M; // This is p'_0 from eq. (28b)
    //const double pmag = sqrt( ( -1* t ) * ( 1. - (t / (4. * M * M ))) ); // p' magnitude from eq. (28b)
    //Angular Kinematics of outgoing photon
    const double cth = -1. / sqrt( 1. + gg ) * ( 1. + gg / 2. * ( 1. + t / QQ ) / ( 1. + x * t / QQ ) ); // This is cos(theta) eq. (26)
    const double theta = acos(cth); // theta angle
    //print('Theta: ', theta)
    //Lepton Angle Kinematics of initial lepton
    const double sthl = sqrt( gg ) / sqrt( 1. + gg ) * ( sqrt ( 1. - y - y * y * gg / 4. ) ); // sin(theta_l) from eq. (22a)
    const double cthl = -1. / sqrt( 1. + gg ) * ( 1. + y * gg / 2. );  // cos(theta_l) from eq. (22a)
    //ratio of momentum transfer to proton mass
    self->tau = -0.25 * t / M2;

    // phi independent 4 - momenta vectors defined on eq. (21) -------------
    const TLorentzVector K = TLorentzVector_Init(k * sthl, 0.0, k * cthl, k);
    const TLorentzVector KP = TLorentzVector_Init(K.x, 0.0, k * ( cthl + y * sqrt( 1. + gg ) ), kpr);
    const TLorentzVector Q = TLorentzVector_minus(K, KP);
    const TLorentzVector p = TLorentzVector_Init(0.0, 0.0, 0.0, M);

    // Sets the Mandelstam variable s which is the center of mass energy
    const double s = TLorentzVector_mul(TLorentzVector_plus(p, K), TLorentzVector_plus(p, K));

    // The Gamma factor in front of the cross section
    self->Gamma = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / ( s - M2 ) / ( s - M2 ) / sqrt( 1. + gg ) / x;

    // Defurne's Jacobian
    const double jcob = 1./ ( 2. * M * x * K.t ) * 2. * PI * 2.;



    // Set 4 Vector Products, phi-independent

    // 4-vectors products (phi - independent)
    const double kkp  = TLorentzVector_mul(K, KP);   //(kk')
    //const double kq   = TLorentzVector_mul(K, Q);    //(kq)
    //const double kp   = TLorentzVector_mul(K, p);    //(pk)
    //const double kpp  = TLorentzVector_mul(KP, p);   //(pk')


    self->phiValues = (double *) malloc(sizeof(double) * numPhiValues);
    self->xbhUU = (double *) malloc(sizeof(double) * numPhiValues);
    self->con_AUUI = (double *) malloc(sizeof(double) * numPhiValues);
    self->con_BUUI = (double *) malloc(sizeof(double) * numPhiValues);
    self->con_CUUI = (double *) malloc(sizeof(double) * numPhiValues);

    for (unsigned long i = 0; i < numPhiValues; i++) {
        self->phiValues[i] = phiValues[i];




        // Set4VectorsPhiDep

        const TLorentzVector QP = TLorentzVector_Init(qp * sin(theta) * cos( phiValues[i] * RAD ), qp * sin(theta) * sin( phiValues[i] * RAD ), qp * cos(theta), qp);
        const TLorentzVector D = TLorentzVector_minus(Q, QP); // delta vector eq. (12a)
        //print(D, "\n", Q, "\n", QP)
        const TLorentzVector pp = TLorentzVector_plus(p, D); // p' from eq. (21)
        TLorentzVector P = TLorentzVector_plus(p, pp);
        P = TLorentzVector_Init(.5 * P.x, .5 * P.y, .5 * P.z, .5 * P.t);




        // Set 4 Vector Products, phi-dependent

        // 4-vectors products (phi - dependent)
        const double kd   = TLorentzVector_mul(K, D);    //(kd)
        const double kpd  = TLorentzVector_mul(KP, D);   //(k'd)
        const double kP   = TLorentzVector_mul(K, P);    //(kP)
        const double kpP  = TLorentzVector_mul(KP, P);   //(k'P)
        const double kqp  = TLorentzVector_mul(K, QP);   //(kq')
        const double kpqp = TLorentzVector_mul(KP, QP);  //(k'q')
        //const double dd   = TLorentzVector_mul(D, D);    //(dd)
        const double Pq   = TLorentzVector_mul(P, Q);    //(Pq)
        const double Pqp  = TLorentzVector_mul(P, QP);   //(Pq')
        const double qd   = TLorentzVector_mul(Q, D);    //(qd)
        const double qpd  = TLorentzVector_mul(QP, D);   //(q'd)

        // //Transverse vector products defined after eq.(241c) -----------------------
        const double kk_T   = BHDVCS_TProduct(&K, &K);
        const double kkp_T  = kk_T;
        const double kqp_T  = BHDVCS_TProduct(&K, &QP);
        const double kd_T   = -1.* kqp_T;
        const double dd_T   = BHDVCS_TProduct(&D, &D);
        //const double kpqp_T = BHDVCS_TProduct(&KP, &QP);
        const double kP_T   = BHDVCS_TProduct(&K, &P);
        const double kpP_T  = BHDVCS_TProduct(&KP, &P);
        const double qpP_T  = BHDVCS_TProduct(&QP, &P);
        //const double kpd_T  = BHDVCS_TProduct(&KP, &D);
        //const double qpd_T  = BHDVCS_TProduct(&QP, &D);




        // Get BHUUxs

        // Coefficients of the BH unpolarized structure function FUUBH
        const double AUUBH = ( (8. * M2) / (t * kqp * kpqp) ) * ( (4. * self->tau * (kP * kP + kpP * kpP) ) - ( (self->tau + 1.) * (kd * kd + kpd * kpd) ) );
        const double BUUBH = ( (16. * M2) / (t* kqp * kpqp) ) * (kd * kd + kpd * kpd);

        // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
        // I multiply by 2 because I think Auu and Buu are missing a factor 2
        const double con_AUUBH = 2. * AUUBH * GeV2nb * jcob;
        const double con_BUUBH = 2. * BUUBH * GeV2nb * jcob;

        // Unpolarized Coefficients multiplied by the Form Factors
        const double bhAUU = (self->Gamma/t) * con_AUUBH * ( F1 * F1 + self->tau * F2 * F2 );
        const double bhBUU = (self->Gamma/t) * con_BUUBH * ( self->tau * ( F1 + F2 ) * ( F1 + F2 ) );

        // Unpolarized BH cross section
        self->xbhUU[i] = bhAUU + bhBUU;


        // first half of Get IUUxs

        // Interference coefficients given on eq. (241a,b,c)--------------------
        const double AUUI = -4.0 / (kqp * kpqp) * (( QQ + t ) * ( 2.0 * ( kP + kpP ) * kk_T + ( Pq * kqp_T ) + 2.* ( kpP * kqp ) - 2.* ( kP * kpqp ) + kpqp * kP_T + kqp * kpP_T - 2. * kkp * kP_T )
            + ( QQ - t + 4.* kd ) * ( Pqp * ( kkp_T + kqp_T - 2.* kkp ) + 2.* kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) );

        const double BUUI = 2.0 * xi / ( kqp * kpqp) * ( ( QQ + t ) * ( 2.* kk_T * ( kd + kpd ) + kqp_T * ( qd - kqp - kpqp + 2. * kkp ) + 2.* kqp * kpd - 2.* kpqp * kd ) +
                                                    ( QQ - t + 4.* kd ) * ( ( kk_T - 2.* kkp ) * qpd - kkp * dd_T - 2.* kd_T * kqp ) );
        const double CUUI = 2.0 / ( kqp * kpqp) * ( -1. * ( QQ + t ) * ( 2.* kkp - kpqp - kqp + 2.* xi * (2.* kkp * kP_T - kpqp * kP_T - kqp * kpP_T) ) * kd_T +
                                                    ( QQ - t + 4.* kd ) * ( ( kqp + kpqp ) * kd_T + dd_T * kkp + 2.* xi * ( kkp * qpP_T - kpqp * kP_T - kqp * kpP_T ) ) );

        // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian

        //print(AUUI, GeV2nb, jcob)
        self->con_AUUI[i] = AUUI * GeV2nb * jcob;
        self->con_BUUI[i] = BUUI * GeV2nb * jcob;
        self->con_CUUI[i] = CUUI * GeV2nb * jcob;
    }

}


double BHDVCS_TProduct(const TLorentzVector * __restrict__ v1, const TLorentzVector * __restrict__ v2) {
    return (v1->x * v2->x) + (v1->y * v2->y);
}


double BHDVCS_GetIUUxs(const BHDVCS * __restrict__ self, double phi, double ReH, double ReE, double ReHtilde) {
    const unsigned long i = BHDVCS_getPhiIndex(self, phi);

    //Unpolarized Coefficients multiplied by the Form Factors
    const double iAUU = (self->Gamma/(-self->t * self->QQ)) * cos( phi * RAD ) * self->con_AUUI[i] * ( self->F1 * ReH + self->tau * self->F2 * ReE );
    const double iBUU = (self->Gamma/(-self->t * self->QQ)) * cos( phi * RAD ) * self->con_BUUI[i] * ( self->F1 + self->F2 ) * ( ReH + ReE );
    const double iCUU = (self->Gamma/(-self->t * self->QQ)) * cos( phi * RAD ) * self->con_CUUI[i] * ( self->F1 + self->F2 ) * ReHtilde;

    // Unpolarized BH-DVCS interference cross section
    return iAUU + iBUU + iCUU; // return xIUU
}

double BHDVCS_getBHUU_plus_getIUU(const BHDVCS * const __restrict__ self, const double phi, const double ReH, const double ReE, const double ReHtilde) {
    const unsigned long i = BHDVCS_getPhiIndex(self, phi);

    return self->xbhUU[i] + BHDVCS_GetIUUxs(self, phi, ReH, ReE, ReHtilde);
}

void BHDVCS_Destruct(const BHDVCS * const __restrict__ self) {
    free(self->phiValues);
    free(self->con_AUUI);
    free(self->con_BUUI);
    free(self->con_CUUI);
    free(self->xbhUU);
}

#endif // BHDVCS_H_
