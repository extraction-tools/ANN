#ifndef TVA1_UU_H
#define TVA1_UU_H

#define TVA1_UU_ALP_INV 137.0359998 // 1 / Electromagnetic Fine Structure Constant
#define TVA1_UU_PI 3.1415926535
#define TVA1_UU_RAD (TVA1_UU_PI / 180.)
#define TVA1_UU_M 0.938272 //Mass of the proton in GeV
#define TVA1_UU_GeV2nb 389379 // Conversion from GeV to NanoBarn = .389379*1000000

#include <math.h>

#include "LorentzVector.h"

static double TVA1_UU_QQ, TVA1_UU_x, TVA1_UU_t, TVA1_UU_k;
static double TVA1_UU_y, TVA1_UU_e, TVA1_UU_xi, TVA1_UU_tmin, TVA1_UU_kpr, TVA1_UU_gg, TVA1_UU_q, TVA1_UU_qp, TVA1_UU_po, TVA1_UU_pmag, TVA1_UU_cth, TVA1_UU_theta, TVA1_UU_sth, TVA1_UU_sthl, TVA1_UU_cthl, TVA1_UU_cthpr, TVA1_UU_sthpr, TVA1_UU_M2, TVA1_UU_tau;
static double TVA1_UU_s;     // Mandelstam variable s which is the center of mass energy
static double TVA1_UU_Gamma; // Factor in front of the cross section
static double TVA1_UU_jcob;  //Defurne's Jacobian

// 4-momentum vectors
static LorentzVector TVA1_UU_K, TVA1_UU_KP, TVA1_UU_Q, TVA1_UU_QP, TVA1_UU_D, TVA1_UU_p, TVA1_UU_P;
// 4 - vector products independent of phi
static double TVA1_UU_kkp, TVA1_UU_kq, TVA1_UU_kp, TVA1_UU_kpp;
// 4 - vector products dependent of phi
static double TVA1_UU_kd, TVA1_UU_kpd, TVA1_UU_kP, TVA1_UU_kpP, TVA1_UU_kqp, TVA1_UU_kpqp, TVA1_UU_dd, TVA1_UU_Pq, TVA1_UU_Pqp, TVA1_UU_qd, TVA1_UU_qpd;
// Transverse 4-vector products
static double TVA1_UU_kk_T, TVA1_UU_kqp_T, TVA1_UU_kkp_T, TVA1_UU_kd_T, TVA1_UU_dd_T, TVA1_UU_kpqp_T, TVA1_UU_kP_T, TVA1_UU_kpP_T, TVA1_UU_qpP_T, TVA1_UU_kpd_T, TVA1_UU_qpd_T;
//Expressions that appear in the polarized interference coefficient calculations
static double TVA1_UU_Dplus, TVA1_UU_Dminus;

static double TVA1_UU_FUUT;
static double TVA1_UU_AUUBH, TVA1_UU_BUUBH; // Coefficients of the BH unpolarized structure function FUU_BH
static double TVA1_UU_AUUI, TVA1_UU_BUUI, TVA1_UU_CUUI; // Coefficients of the BHDVCS interference unpolarized structure function FUU_I
static double TVA1_UU_con_AUUBH, TVA1_UU_con_BUUBH; // Coefficients times the conversion to nb and the jacobian
static double TVA1_UU_con_AUUI, TVA1_UU_con_BUUI, TVA1_UU_con_CUUI;  // Coefficients times the conversion to nb and the jacobian
static double TVA1_UU_bhAUU, TVA1_UU_bhBUU; // Auu and Buu term of the BH cross section
static double TVA1_UU_iAUU, TVA1_UU_iBUU, TVA1_UU_iCUU; // Terms of the interference containing AUUI, BUUI and CUUI
static double TVA1_UU_xdvcsUU, TVA1_UU_xbhUU, TVA1_UU_xIUU; // Unpolarized cross sections



void TVA1_UU_Init(void);
double TVA1_UU_TProduct(LorentzVector v1, LorentzVector v2);
void TVA1_UU_SetKinematics(double _QQ, double _x, double _t, double _k);
void TVA1_UU_Set4VectorsPhiDep(double phi);
void TVA1_UU_Set4VectorProducts(void);
double TVA1_UU_GetDVCSUU(double ReH, double ReE, double ReHtilde, double ReEtilde, double ImH, double ImE, double ImHtilde, double ImEtilde);
double TVA1_UU_GetBHUU(double phi, double F1, double F2);
double TVA1_UU_GetIUU(double phi, double F1, double F2, double ReH, double ReE, double ReHtilde);



double TVA1_UU_TProduct(LorentzVector v1, LorentzVector v2) {
    // Transverse product (tv1v2)
    return v1.x * v2.x + v1.y * v2.y;
}



void TVA1_UU_SetKinematics(double _QQ, double _x, double _t, double _k){

    TVA1_UU_QQ = _QQ; //Q^2 value
    TVA1_UU_x = _x;   //Bjorken x
    TVA1_UU_t = _t;   //momentum transfer squared
    TVA1_UU_k = _k;   //Electron Beam Energy
    TVA1_UU_M2 = TVA1_UU_M*TVA1_UU_M; //Mass of the proton  squared in GeV
    //fractional energy of virtual photon
    TVA1_UU_y = TVA1_UU_QQ / ( 2. * TVA1_UU_M * TVA1_UU_k * TVA1_UU_x ); // From eq. (23) where gamma is substituted from eq (12c)
    //squared gamma variable ratio of virtuality to energy of virtual photon
    TVA1_UU_gg = 4. * TVA1_UU_M2 * TVA1_UU_x * TVA1_UU_x / TVA1_UU_QQ; // This is gamma^2 [from eq. (12c)]
    //ratio of longitudinal to transverse virtual photon flux
    TVA1_UU_e = ( 1 - TVA1_UU_y - ( TVA1_UU_y * TVA1_UU_y * (TVA1_UU_gg / 4.) ) ) / ( 1. - TVA1_UU_y + (TVA1_UU_y * TVA1_UU_y / 2.) + ( TVA1_UU_y * TVA1_UU_y * (TVA1_UU_gg / 4.) ) ); // epsilon eq. (32)
    //Skewness parameter
    TVA1_UU_xi = 1. * TVA1_UU_x * ( ( 1. + TVA1_UU_t / ( 2. * TVA1_UU_QQ ) ) / ( 2. - TVA1_UU_x + TVA1_UU_x * TVA1_UU_t / TVA1_UU_QQ ) ); // skewness parameter eq. (12b) note: there is a minus sign on the write up that shouldn't be there
    //xi = x * ( ( 1. ) / ( 2. - x ) );
    //Minimum t value
    TVA1_UU_tmin = -( TVA1_UU_QQ * ( 1. - sqrt( 1. + TVA1_UU_gg ) + TVA1_UU_gg / 2. ) ) / ( TVA1_UU_x * ( 1. - sqrt( 1. + TVA1_UU_gg ) + TVA1_UU_gg / ( 2.* TVA1_UU_x ) ) ); // minimum t eq. (29)
    //Final Lepton energy
    TVA1_UU_kpr = TVA1_UU_k * ( 1. - TVA1_UU_y ); // k' from eq. (23)
    //outgoing photon energy
    TVA1_UU_qp = TVA1_UU_t / 2. / TVA1_UU_M + TVA1_UU_k - TVA1_UU_kpr; //q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
    //Final proton Energy
    TVA1_UU_po = TVA1_UU_M - TVA1_UU_t / 2. / TVA1_UU_M; // This is p'_0 from eq. (28b)
    TVA1_UU_pmag = sqrt( ( -TVA1_UU_t ) * ( 1. - TVA1_UU_t / 4. / TVA1_UU_M / TVA1_UU_M ) ); // p' magnitude from eq. (28b)
    //Angular Kinematics of outgoing photon
    TVA1_UU_cth = -1. / sqrt( 1. + TVA1_UU_gg ) * ( 1. + TVA1_UU_gg / 2. * ( 1. + TVA1_UU_t / TVA1_UU_QQ ) / ( 1. + TVA1_UU_x * TVA1_UU_t / TVA1_UU_QQ ) ); // This is cos(theta) eq. (26)
    TVA1_UU_theta = acos(TVA1_UU_cth); // theta angle
    //Lepton Angle Kinematics of initial lepton
    TVA1_UU_sthl = sqrt( TVA1_UU_gg ) / sqrt( 1. + TVA1_UU_gg ) * ( sqrt ( 1. - TVA1_UU_y - TVA1_UU_y * TVA1_UU_y * TVA1_UU_gg / 4. ) ); // sin(theta_l) from eq. (22a)
    TVA1_UU_cthl = -1. / sqrt( 1. + TVA1_UU_gg ) * ( 1. + TVA1_UU_y * TVA1_UU_gg / 2. ) ; // cos(theta_l) from eq. (22a)
    //ratio of momentum transfer to proton mass
    TVA1_UU_tau = -0.25 * TVA1_UU_t / TVA1_UU_M2;

    // phi independent 4 - momenta vectors defined on eq. (21) -------------
    TVA1_UU_K = LorentzVector_SetPxPyPzE( TVA1_UU_k * TVA1_UU_sthl, 0.0, TVA1_UU_k * TVA1_UU_cthl, TVA1_UU_k );
    TVA1_UU_KP = LorentzVector_SetPxPyPzE( TVA1_UU_K.x, 0.0, TVA1_UU_k * ( TVA1_UU_cthl + TVA1_UU_y * sqrt( 1. + TVA1_UU_gg ) ), TVA1_UU_kpr );
    TVA1_UU_Q = LorentzVector_sub(TVA1_UU_K, TVA1_UU_KP);
    TVA1_UU_p = LorentzVector_SetPxPyPzE(0.0, 0.0, 0.0, TVA1_UU_M);

    // Sets the Mandelstam variable s which is the center of mass energy
    TVA1_UU_s = LorentzVector_mul(LorentzVector_add(TVA1_UU_p, TVA1_UU_K), LorentzVector_add(TVA1_UU_p, TVA1_UU_K));

    // The Gamma factor in front of the cross section
    TVA1_UU_Gamma = 1. / TVA1_UU_ALP_INV / TVA1_UU_ALP_INV / TVA1_UU_ALP_INV / TVA1_UU_PI / TVA1_UU_PI / 16. / ( TVA1_UU_s - TVA1_UU_M2 ) / ( TVA1_UU_s - TVA1_UU_M2 ) / sqrt( 1. + TVA1_UU_gg ) / TVA1_UU_x;

    // Defurne's Jacobian
    TVA1_UU_jcob = 2. * TVA1_UU_PI;
}



void TVA1_UU_Set4VectorsPhiDep(double phi) {
    // phi dependent 4 - momenta vectors defined on eq. (21) -------------
    TVA1_UU_QP = LorentzVector_SetPxPyPzE(TVA1_UU_qp * sin(TVA1_UU_theta) * cos( phi * TVA1_UU_RAD ), TVA1_UU_qp * sin(TVA1_UU_theta) * sin( phi * TVA1_UU_RAD ), TVA1_UU_qp * cos(TVA1_UU_theta), TVA1_UU_qp);
    TVA1_UU_D = LorentzVector_sub(TVA1_UU_Q, TVA1_UU_QP); // delta vector eq. (12a)
    LorentzVector pp = LorentzVector_add(TVA1_UU_p, TVA1_UU_D); // p' from eq. (21)
    TVA1_UU_P = LorentzVector_add(TVA1_UU_p, pp);
    TVA1_UU_P = LorentzVector_SetPxPyPzE(.5 * TVA1_UU_P.x, .5 * TVA1_UU_P.y, .5 * TVA1_UU_P.z, .5 * TVA1_UU_P.t);
}



void TVA1_UU_Set4VectorProducts(void) {
    // 4-vectors products (phi - independent)
    TVA1_UU_kkp  = LorentzVector_mul(TVA1_UU_K, TVA1_UU_KP);   //(kk')
    TVA1_UU_kq   = LorentzVector_mul(TVA1_UU_K,  TVA1_UU_Q);    //(kq)
    TVA1_UU_kp   = LorentzVector_mul(TVA1_UU_K, TVA1_UU_p);    //(pk)
    TVA1_UU_kpp  = LorentzVector_mul(TVA1_UU_KP, TVA1_UU_p);   //(pk')

    // 4-vectors products (phi - dependent)
    TVA1_UU_kd   = LorentzVector_mul(TVA1_UU_K, TVA1_UU_D);    //(kΔ)
    TVA1_UU_kpd  = LorentzVector_mul(TVA1_UU_KP, TVA1_UU_D);   //(k'Δ)
    TVA1_UU_kP   = LorentzVector_mul(TVA1_UU_K, TVA1_UU_P);    //(kP)
    TVA1_UU_kpP  = LorentzVector_mul(TVA1_UU_KP, TVA1_UU_P);   //(k'P)
    TVA1_UU_kqp  = LorentzVector_mul(TVA1_UU_K, TVA1_UU_QP);   //(kq')
    TVA1_UU_kpqp = LorentzVector_mul(TVA1_UU_KP, TVA1_UU_QP);  //(k'q')
    TVA1_UU_dd   = LorentzVector_mul(TVA1_UU_D, TVA1_UU_D);    //(ΔΔ)
    TVA1_UU_Pq   = LorentzVector_mul(TVA1_UU_P, TVA1_UU_Q);    //(Pq)
    TVA1_UU_Pqp  = LorentzVector_mul(TVA1_UU_P, TVA1_UU_QP);   //(Pq')
    TVA1_UU_qd   = LorentzVector_mul(TVA1_UU_Q, TVA1_UU_D);    //(q
    TVA1_UU_qpd  = LorentzVector_mul(TVA1_UU_QP, TVA1_UU_D);   //(q'Δ)

    // Transverse vector products defined after eq.(241c) -----------------------
    TVA1_UU_kk_T   = TVA1_UU_TProduct(TVA1_UU_K, TVA1_UU_K);
    TVA1_UU_kkp_T  = TVA1_UU_kk_T;
    TVA1_UU_kqp_T  = TVA1_UU_TProduct(TVA1_UU_K, TVA1_UU_QP);
    TVA1_UU_kd_T   = -1.* TVA1_UU_kqp_T;
    TVA1_UU_dd_T   = TVA1_UU_TProduct(TVA1_UU_D, TVA1_UU_D);
    TVA1_UU_kpqp_T = TVA1_UU_kqp_T;
    TVA1_UU_kP_T   = TVA1_UU_TProduct(TVA1_UU_K, TVA1_UU_P);
    TVA1_UU_kpP_T  = TVA1_UU_TProduct(TVA1_UU_KP, TVA1_UU_P);
    TVA1_UU_qpP_T  = TVA1_UU_TProduct(TVA1_UU_QP, TVA1_UU_P);
    TVA1_UU_kpd_T  = -1.* TVA1_UU_kqp_T;
    TVA1_UU_qpd_T  = -1. * TVA1_UU_dd_T;

    //Expressions that appear in the interference coefficient calculations
    TVA1_UU_Dplus   = .5 / TVA1_UU_kpqp - .5 / TVA1_UU_kqp;
    TVA1_UU_Dminus  = -.5 / TVA1_UU_kpqp - .5 / TVA1_UU_kqp;
}



//====================================================================================
// DVCS Unpolarized Cross Section
//====================================================================================
double TVA1_UU_GetDVCSUU(double ReH, double ReE, double ReHtilde, double ReEtilde, double ImH, double ImE, double ImHtilde, double ImEtilde) {
    // Note: SetKinematics should have been previously called.
    TVA1_UU_FUUT = 4.* ( ( 1 - TVA1_UU_xi * TVA1_UU_xi ) * ( ReH * ReH + ImH * ImH + ReHtilde * ReHtilde + ImHtilde * ImHtilde ) + ( TVA1_UU_tmin - TVA1_UU_t ) / ( 2.* TVA1_UU_M2 ) * ( ReE * ReE + ImE * ImE + TVA1_UU_xi * TVA1_UU_xi * ReEtilde * ReEtilde + TVA1_UU_xi * TVA1_UU_xi * ImEtilde * ImEtilde )
        - ( 2.* TVA1_UU_xi * TVA1_UU_xi ) / ( 1 - TVA1_UU_xi * TVA1_UU_xi ) * ( ReH * ReE + ImH * ImE + ReHtilde * ReEtilde + ImHtilde * ImEtilde ) );

    TVA1_UU_xdvcsUU =  TVA1_UU_GeV2nb * TVA1_UU_jcob * TVA1_UU_Gamma / TVA1_UU_QQ / ( 1 - TVA1_UU_e ) * TVA1_UU_FUUT;

    return TVA1_UU_xdvcsUU;
}



//====================================================================================
// BH Unpolarized Cross Section
//====================================================================================
double TVA1_UU_GetBHUU(double phi, double F1, double F2) {
    // Get the 4-vector products. Note: SetKinematics should have been previously called.
    TVA1_UU_Set4VectorsPhiDep(phi);
    TVA1_UU_Set4VectorProducts();

    // Coefficients of the BH unpolarized structure function FUU_BH
    TVA1_UU_AUUBH = (8. * TVA1_UU_M2) / (TVA1_UU_t * TVA1_UU_kqp * TVA1_UU_kpqp) * ( (4. * TVA1_UU_tau * (TVA1_UU_kP * TVA1_UU_kP + TVA1_UU_kpP * TVA1_UU_kpP) ) - ( (TVA1_UU_tau + 1.) * (TVA1_UU_kd * TVA1_UU_kd + TVA1_UU_kpd * TVA1_UU_kpd) ) );  //eq. 147
    TVA1_UU_BUUBH = (16. * TVA1_UU_M2) / (TVA1_UU_t * TVA1_UU_kqp * TVA1_UU_kpqp) * (TVA1_UU_kd * TVA1_UU_kd + TVA1_UU_kpd * TVA1_UU_kpd); // eq. 148

    // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
    TVA1_UU_con_AUUBH = TVA1_UU_AUUBH * TVA1_UU_GeV2nb * TVA1_UU_jcob;
    TVA1_UU_con_BUUBH = TVA1_UU_BUUBH * TVA1_UU_GeV2nb * TVA1_UU_jcob;

    // Unpolarized Coefficients multiplied by the Form Factors
    TVA1_UU_bhAUU = (TVA1_UU_Gamma/TVA1_UU_t) * TVA1_UU_con_AUUBH * ( F1 * F1 + TVA1_UU_tau * F2 * F2 );
    TVA1_UU_bhBUU = (TVA1_UU_Gamma/TVA1_UU_t) * TVA1_UU_con_BUUBH * ( TVA1_UU_tau * ( F1 + F2 ) * ( F1 + F2 ) );

    // Unpolarized BH cross section
    TVA1_UU_xbhUU = TVA1_UU_bhAUU + TVA1_UU_bhBUU;

    return TVA1_UU_xbhUU;
}



//====================================================================================
// Unpolarized BH-DVCS Interference Cross Section (Comparison paper)
//====================================================================================
double TVA1_UU_GetIUU(double phi, double F1, double F2, double ReH, double ReE, double ReHtilde) {
    // Get the 4-vector products. Note: SetKinematics should have been previously called.
    TVA1_UU_Set4VectorsPhiDep(phi);
    TVA1_UU_Set4VectorProducts();

    TVA1_UU_AUUI = -4. * cos( phi * TVA1_UU_RAD ) * ( TVA1_UU_Dplus * ( ( TVA1_UU_kqp_T - 2. * TVA1_UU_kk_T - 2. * TVA1_UU_kqp ) * TVA1_UU_kpP + ( 2. * TVA1_UU_kpqp - 2. * TVA1_UU_kkp_T - TVA1_UU_kpqp_T ) * TVA1_UU_kP + TVA1_UU_kpqp * TVA1_UU_kP_T + TVA1_UU_kqp * TVA1_UU_kpP_T - 2.*TVA1_UU_kkp * TVA1_UU_kP_T ) -
                    TVA1_UU_Dminus * ( ( 2. * TVA1_UU_kkp - TVA1_UU_kpqp_T - TVA1_UU_kkp_T ) * TVA1_UU_Pqp + 2. * TVA1_UU_kkp * TVA1_UU_qpP_T - TVA1_UU_kpqp * TVA1_UU_kP_T - TVA1_UU_kqp * TVA1_UU_kpP_T ) ) ;

    TVA1_UU_BUUI = -2. * TVA1_UU_xi * cos( phi * TVA1_UU_RAD ) * ( TVA1_UU_Dplus * ( ( TVA1_UU_kqp_T - 2. * TVA1_UU_kk_T - 2. * TVA1_UU_kqp ) * TVA1_UU_kpd + ( 2. * TVA1_UU_kpqp - 2. * TVA1_UU_kkp_T - TVA1_UU_kpqp_T ) * TVA1_UU_kd + TVA1_UU_kpqp * TVA1_UU_kd_T + TVA1_UU_kqp * TVA1_UU_kpd_T - 2.*TVA1_UU_kkp * TVA1_UU_kd_T ) -
                        TVA1_UU_Dminus * ( ( 2. * TVA1_UU_kkp - TVA1_UU_kpqp_T - TVA1_UU_kkp_T ) * TVA1_UU_qpd + 2. * TVA1_UU_kkp * TVA1_UU_qpd_T - TVA1_UU_kpqp * TVA1_UU_kd_T - TVA1_UU_kqp * TVA1_UU_kpd_T ) );

    TVA1_UU_CUUI = -2. * cos( phi * TVA1_UU_RAD ) * ( TVA1_UU_Dplus * ( 2. * TVA1_UU_kkp * TVA1_UU_kd_T - TVA1_UU_kpqp * TVA1_UU_kd_T - TVA1_UU_kqp * TVA1_UU_kpd_T + 4. * TVA1_UU_xi * TVA1_UU_kkp * TVA1_UU_kP_T - 2. * TVA1_UU_xi * TVA1_UU_kpqp * TVA1_UU_kP_T - 2. * TVA1_UU_xi * TVA1_UU_kqp * TVA1_UU_kpP_T ) -
                    TVA1_UU_Dminus * ( TVA1_UU_kkp * TVA1_UU_qpd_T - TVA1_UU_kpqp * TVA1_UU_kd_T - TVA1_UU_kqp * TVA1_UU_kpd_T + 2. * TVA1_UU_xi * TVA1_UU_kkp * TVA1_UU_qpP_T - 2. * TVA1_UU_xi * TVA1_UU_kpqp * TVA1_UU_kP_T - 2. * TVA1_UU_xi * TVA1_UU_kqp * TVA1_UU_kpP_T ) );

    // Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
    TVA1_UU_con_AUUI = TVA1_UU_AUUI * TVA1_UU_GeV2nb * TVA1_UU_jcob;
    TVA1_UU_con_BUUI = TVA1_UU_BUUI * TVA1_UU_GeV2nb * TVA1_UU_jcob;
    TVA1_UU_con_CUUI = TVA1_UU_CUUI * TVA1_UU_GeV2nb * TVA1_UU_jcob;

    //Unpolarized Coefficients multiplied by the Form Factors
    TVA1_UU_iAUU = (TVA1_UU_Gamma/(fabs(TVA1_UU_t) * TVA1_UU_QQ)) * TVA1_UU_con_AUUI * ( F1 * ReH + TVA1_UU_tau * F2 * ReE );
    TVA1_UU_iBUU = (TVA1_UU_Gamma/(fabs(TVA1_UU_t) * TVA1_UU_QQ)) * TVA1_UU_con_BUUI * ( F1 + F2 ) * ( ReH + ReE );
    TVA1_UU_iCUU = (TVA1_UU_Gamma/(fabs(TVA1_UU_t) * TVA1_UU_QQ)) * TVA1_UU_con_CUUI * ( F1 + F2 ) * ReHtilde;

    // Unpolarized BH-DVCS interference cross section
    TVA1_UU_xIUU = TVA1_UU_iAUU + TVA1_UU_iBUU + TVA1_UU_iCUU;

    return -1. * TVA1_UU_xIUU;
}

#endif
