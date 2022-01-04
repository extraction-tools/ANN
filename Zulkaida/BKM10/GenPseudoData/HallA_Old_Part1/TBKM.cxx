//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Calculation of unpolarized DVCS cross section using the BKM formulation:    //
//                                                                              //
//  - BKM 2002 cross sections (arXiv:hep-ph/0112108v2)                          //
//      -- BH, DVCDS_UU_02, I_UU_02                                             //
//                                                                              //
//  - BKM 2010 cross sections (arXiv:hep-ph/1005.5209v1)                        //
//      -- DVCS_UU_10, I_UU_10                                                  //
//                                                                              //
//  Written by: Liliet Calero Diaz                                              //
//                                                                              //
//  Email: lc2fc@virginia.edu                                                   //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////


#include "TBKM.h"

using namespace std;  	// std namespace: so you can do things like 'cout'

ClassImp(TBKM)			// classimp: necessary for root


//_______________________________________________________________________________________________________________________________
TBKM::TBKM() {
	// Default Constructor
}
//_______________________________________________________________________________________________________________________________
TBKM::~TBKM() {
	// Default Destructor
}
//_______________________________________________________________________________________________________________________________
TComplex TBKM::cdstar( TComplex c, TComplex d ){ // ( C D* ) product

    TComplex dstar = TComplex::Conjugate(d);

    return ( c.Re() * dstar.Re() - c.Im() * dstar.Im() ) + ( c.Re() * dstar.Im() + c.Im() * dstar.Re() ) * TComplex::I();
}
//_______________________________________________________________________________________________________________________________
void TBKM::SetCFFs( TComplex *t2cffs ) { // Twist-2 Compton Form Factors

     H = t2cffs[0];
     E = t2cffs[1];
     Htilde = t2cffs[2];
     Etilde = t2cffs[3];
}
//_______________________________________________________________________________________________________________________________
void TBKM::SetKinematics( Double_t *kine ) {

    QQ = kine[0];     //Q^2 value
    x = kine[1];      //Bjorken x
    t = kine[2];      //momentum transfer squared
    k = kine[3];      //Electron Beam energy

    ee = 4. * M2 * x * x / QQ; // epsilon squared
    y = sqrt(QQ) / ( sqrt(ee) * k );  // lepton energy fraction
    xi = x * ( 1. + t / 2. / QQ ) / ( 2. - x + x * t / QQ ); // Generalized Bjorken variable
    s = 2. * M * k + M2; // Mandelstan variable
    Gamma = x * y * y / ALP_INV / ALP_INV / ALP_INV / PI / 8. / QQ / QQ / sqrt( 1. + ee ); // factor in front of the cross section, eq. (22)
    tmin = - QQ * ( 2. * ( 1. - x ) * ( 1. - sqrt(1. + ee) ) + ee ) / ( 4. * x * ( 1. - x ) + ee ); // eq. (31)
    Ktilde_10 = sqrt( tmin - t ) * sqrt( ( 1. - x ) * sqrt( 1. + ee ) + ( ( t - tmin ) * ( ee + 4. * x * ( 1. - x ) ) / 4. / QQ ) ) * sqrt( 1. - y - y * y * ee / 4. )
                / sqrt( 1. - y + y * y * ee / 4.); // K tilde from 2010 paper
    K = sqrt( 1. - y + y * y * ee / 4.) * Ktilde_10 / sqrt(QQ);
}
//_______________________________________________________________________________________________________________________________
void TBKM::BHLeptonPropagators(Double_t *kine, Double_t phi) {

    SetKinematics(kine);
    // K*D 4-vector product (phi-dependent)
    KD = - QQ / ( 2. * y * ( 1. + ee ) ) * ( 1. + 2. * K * cos( PI - ( phi * RAD ) ) - t / QQ * ( 1. - x * ( 2. - y ) + y * ee / 2. ) + y * ee / 2.  ); // eq. (29)

    // lepton BH propagators P1 and P2 (contaminating phi-dependence)
    P1 = 1. + 2. * KD / QQ;
    P2 = t / QQ - 2. * KD / QQ;
}
//_______________________________________________________________________________________________________________________________
Double_t TBKM::BH_UU(Double_t *kine, Double_t phi, Double_t F1, Double_t F2) { // BH Unpolarized Cross Section

    // Sets the kinematics and gets BH propagators
    BHLeptonPropagators(kine, phi);

    // BH unpolarized Fourier harmonics eqs. (35 - 37)
    c0_BH = 8. * K * K * ( ( 2. + 3. * ee ) * ( QQ / t ) * ( F1 * F1  - F2 * F2 * t / ( 4. * M2 ) ) + 2. * x * x * ( F1 + F2 ) * ( F1 + F2 ) )
            + ( 2. - y ) * ( 2. - y ) * ( ( 2. + ee ) * ( ( 4. * x * x * M2 / t ) * ( 1. + t / QQ ) * ( 1. + t / QQ ) + 4. * ( 1. - x ) * ( 1. + x * t / QQ ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) )
            + 4. * x * x * ( x + ( 1. - x + ee / 2. ) * ( 1. - t / QQ ) * ( 1. - t / QQ ) - x * ( 1. - 2. * x ) * t * t / ( QQ * QQ ) ) * ( F1 + F2 ) * ( F1 + F2 ) )
            + 8. * ( 1. + ee ) * ( 1. - y - ee * y * y / 4. ) * ( 2. * ee * ( 1. - t / ( 4. * M2 ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) - x * x * ( 1. - t / QQ ) * ( 1. - t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) );

    c1_BH = 8. * K * ( 2. - y ) * ( ( 4. * x * x * M2 / t - 2. * x - ee ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 2. * x * x * ( 1. - ( 1. - 2. * x ) * t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) );

    c2_BH = 8. * x * x * K * K * ( ( 4. * M2 / t ) * ( F1 * F1 - F2 * F2 * t / ( 4. * M2 ) ) + 2. * ( F1 + F2 ) * ( F1 + F2 ) );

    // BH squared amplitude eq (25) divided by e^6
    Amp2_BH = 1. / ( x * x * y * y * ( 1. + ee ) * ( 1. + ee ) * t * P1 * P2 ) * ( c0_BH + c1_BH * cos( PI - (phi * RAD) ) + c2_BH * cos( 2. * ( PI - ( phi * RAD ) ) )  );

    Amp2_BH = GeV2nb * Amp2_BH; // convertion to nb

    return dsigma_BH = Gamma * Amp2_BH;
}
//===============================================================================================================================
//                   B   K   M   -   2   0   0   2
//===============================================================================================================================
//_______________________________________________________________________________________________________________________________
Double_t TBKM::DVCS_UU_02(Double_t *kine, Double_t phi, TComplex *t2cffs, TString twist = "t2") { // Pure DVCS Unpolarized Cross Section

    SetKinematics(kine);

    SetCFFs(t2cffs);

    // c coefficients (BKM02 eqs. [66]) for pure DVCS
    c_dvcs = 1./(2. - x)/(2. - x) * ( 4. * ( 1 - x ) * ( H.Rho2() + Htilde.Rho2() ) - x * x * ( cdstar(H, E) + cdstar(E, H) + cdstar(Htilde, Etilde) + cdstar(Etilde, Htilde) )
             - ( x * x + (2. - x) * (2. - x) * t / 4. / M2 ) * E.Rho2() - ( x * x * t / 4. / M2 ) * Etilde.Rho2() );

    // Pure DVCS unpolarized Fourier harmonics (BKM02 eqs. [43, 44])
    c0_dvcs = 2. * ( 2. - 2. * y + y * y ) * c_dvcs;
    c1_dvcs = - ( 2. * xi / ( 1. + xi ) ) * ( 8. * K / ( 2. - x ) ) * ( 2. - y ) * c_dvcs;

    // DVCS squared amplitude eq (26) divided by e^6
    if( twist == "t2") // F_eff = 0
        Amp2_DVCS = 1. / ( y * y * QQ ) *  c0_dvcs ;

    if( twist == "t3")  // F_eff = -2xi/(1+xi) F
        Amp2_DVCS = 1. / ( y * y * QQ ) * ( c0_dvcs + c1_dvcs * cos( PI - (phi * RAD) ) );

    Amp2_DVCS = GeV2nb * Amp2_DVCS; // convertion to nb

    return dsigma_DVCS = Gamma * Amp2_DVCS;
}
//_______________________________________________________________________________________________________________________________
Double_t TBKM::I_UU_02(Double_t *kine, Double_t phi, Double_t F1, Double_t F2, TComplex *t2cffs, TString twist = "t2") { // Interference Unpolarized Cross Section (Liuti's style)

    // Get BH propagators and set the kinematics
    BHLeptonPropagators(kine, phi);

    SetCFFs(t2cffs); // Etilde CFF does not appear in the interference

    Double_t A_02, B_02, C_02; // Coefficients in from to the CFFs

    if( twist == "t2") // F_eff = 0, no c2 term (no cos(2phi))
        A_02 = - 8. * K * K * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * ( 2. - y ) * ( 1. - y ) * ( 2. - x ) * t / QQ - 8. * K * ( 2. - 2. * y + y * y ) * cos( PI - (phi * RAD) );

    if( twist == "t3") // F_eff = -2xi/(1+xi) F
        A_02 = - 8. * K * K * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * ( 2. - y ) * ( 1. - y ) * ( 2. - x ) * t / QQ - 8. * K * ( 2. - 2. * y + y * y ) * cos( PI - (phi * RAD) )
               + 32. * K * K * xi * ( 2. - y ) / ( 2. - x ) / ( 1. + xi ) * cos( 2. * ( PI - ( phi * RAD ) ) );

    B_02 = 8. * x * x * ( 2. - y ) * (1 - y ) / ( 2. - x ) * t / QQ;
    // C =  x / ( 2. - x ) * ( - 8. * K * K * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * K * ( 2. - 2. * y + y * y ) * cos( PI - (phi * RAD) ) );
    C_02 = x / ( 2. - x ) * ( A_02 + ( 2. - x ) * ( 2. - x) / x / x * B_02 );

    // BH-DVCS interference squared amplitude eq (27) divided by e^6
    I = 1. / ( x * y * y * y * t * P1 * P2 ) * ( A_02 * ( F1 * H.Re() - t / 4. / M2 * F2 * E.Re() ) + B_02 * ( F1 + F2 ) * ( H.Re() + E.Re() ) + C_02 * ( F1 + F2 ) * Htilde.Re() );

    I = GeV2nb * I; // convertion to nb

    return dsigma_I = Gamma * I;
}
//===============================================================================================================================
//                   B   K   M   -   2   0   1   0
//===============================================================================================================================
//_______________________________________________________________________________________________________________________________
Double_t TBKM::DVCS_UU_10(Double_t *kine, Double_t phi, TComplex *t2cffs, TString twist = "t2") { // Pure DVCS Unpolarized Cross Section

    SetKinematics(kine);

    SetCFFs(t2cffs);

    // F_eff = f * F
    if(twist == "t2") f = 0; // F_eff = 0 --> DVCS is constant
    if(twist == "t3") f = - 2. * xi / ( 1. + xi );
    if(twist == "t3ww") f = 2. / ( 1. + xi );

    // c_dvcs_unp(F,F*) coefficients (BKM10 eqs. [2.22]) for pure DVCS
    c_dvcs_ffs = QQ * ( QQ + x * t ) / pow( ( ( 2. - x ) * QQ + x * t ), 2) * ( 4. * ( 1. - x ) * H.Rho2() + 4. * ( 1. - x + ( 2. * QQ + t ) / ( QQ + x * t ) * ee / 4. ) * Htilde.Rho2()
                 - x * x * pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) * ( cdstar(H, E) + cdstar(E, H) ) - x * x * QQ / ( QQ + x * t ) * ( cdstar(Htilde, Etilde) + cdstar(Etilde, Htilde) )
                 - ( x * x * pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) + pow( ( ( 2. - x ) * QQ + x * t ), 2) / QQ / ( QQ + x * t ) * t / 4. / M2 ) * E.Rho2()
                 - ( x * x * QQ / ( QQ + x * t ) * t / 4. / M2 ) * Etilde.Rho2() );
    // c_dvcs_unp(Feff,Feff*)
    c_dvcs_effeffs = f * f * c_dvcs_ffs;
    // c_dvcs_unp(Feff,F*)
    c_dvcs_efffs = f * c_dvcs_ffs;

    // dvcs c_n coefficients (BKM10 eqs. [2.18], [2.19])
    c0_dvcs_10 = 2. * ( 2. - 2. * y + y * y  + ee / 2. * y * y ) / ( 1. + ee ) * c_dvcs_ffs + 16. * K * K / pow(( 2. - x ), 2) / ( 1. + ee ) * c_dvcs_effeffs;
    c1_dvcs_10 = 8. * K / ( 2. - x ) / ( 1. + ee ) * ( 2. - y ) * c_dvcs_efffs;

    Amp2_DVCS_10 = 1. / ( y * y * QQ ) * ( c0_dvcs_10 + c1_dvcs_10 * cos( PI - (phi * RAD) ) );

    Amp2_DVCS_10 = GeV2nb * Amp2_DVCS_10; // convertion to nb

    return dsigma_DVCS_10 = Gamma * Amp2_DVCS_10;
}
//_______________________________________________________________________________________________________________________________
Double_t TBKM::Get_c0fit(Double_t *kine,  TComplex *t2cffs, TString twist = "t2") { // Pure DVCS Unpolarized Cross Section
    SetKinematics(kine);
    SetCFFs(t2cffs);
    // F_eff = f * F
    if(twist == "t2") f = 0; // F_eff = 0 --> DVCS is constant
    if(twist == "t3") f = - 2. * xi / ( 1. + xi );
    if(twist == "t3ww") f = 2. / ( 1. + xi );
    // c_dvcs_unp(F,F*) coefficients (BKM10 eqs. [2.22]) for pure DVCS
    c_dvcs_ffs = QQ * ( QQ + x * t ) / pow( ( ( 2. - x ) * QQ + x * t ), 2) * ( 4. * ( 1. - x ) * H.Rho2() + 4. * ( 1. - x + ( 2. * QQ + t ) / ( QQ + x * t ) * ee / 4. ) * Htilde.Rho2()
                 - x * x * pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) * ( cdstar(H, E) + cdstar(E, H) ) - x * x * QQ / ( QQ + x * t ) * ( cdstar(Htilde, Etilde) + cdstar(Etilde, Htilde) )
                 - ( x * x * pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) + pow( ( ( 2. - x ) * QQ + x * t ), 2) / QQ / ( QQ + x * t ) * t / 4. / M2 ) * E.Rho2()
                 - ( x * x * QQ / ( QQ + x * t ) * t / 4. / M2 ) * Etilde.Rho2() );
    // c_dvcs_unp(Feff,Feff*)
    c_dvcs_effeffs = f * f * c_dvcs_ffs;
    // c_dvcs_unp(Feff,F*)
    c_dvcs_efffs = f * c_dvcs_ffs;
    // dvcs c_n coefficients (BKM10 eqs. [2.18], [2.19])
    c0_dvcs_10 = 2. * ( 2. - 2. * y + y * y  + ee / 2. * y * y ) / ( 1. + ee ) * c_dvcs_ffs + 16. * K * K / pow(( 2. - x ), 2) / ( 1. + ee ) * c_dvcs_effeffs;
    c1_dvcs_10 = 8. * K / ( 2. - x ) / ( 1. + ee ) * ( 2. - y ) * c_dvcs_efffs;
    
    return DVCS10_c0fit = Gamma * GeV2nb * c0_dvcs_10 * 1. / ( y * y * QQ );
}

Double_t TBKM::Get_c1fit(Double_t *kine,  TComplex *t2cffs, TString twist = "t2") { // Pure DVCS Unpolarized Cross Section
    SetKinematics(kine);
    SetCFFs(t2cffs);
    // F_eff = f * F
    if(twist == "t2") f = 0; // F_eff = 0 --> DVCS is constant
    if(twist == "t3") f = - 2. * xi / ( 1. + xi );
    if(twist == "t3ww") f = 2. / ( 1. + xi );
    // c_dvcs_unp(F,F*) coefficients (BKM10 eqs. [2.22]) for pure DVCS
    c_dvcs_ffs = QQ * ( QQ + x * t ) / pow( ( ( 2. - x ) * QQ + x * t ), 2) * ( 4. * ( 1. - x ) * H.Rho2() + 4. * ( 1. - x + ( 2. * QQ + t ) / ( QQ + x * t ) * ee / 4. ) * Htilde.Rho2()
                 - x * x * pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) * ( cdstar(H, E) + cdstar(E, H) ) - x * x * QQ / ( QQ + x * t ) * ( cdstar(Htilde, Etilde) + cdstar(Etilde, Htilde) )
                 - ( x * x * pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) + pow( ( ( 2. - x ) * QQ + x * t ), 2) / QQ / ( QQ + x * t ) * t / 4. / M2 ) * E.Rho2()
                 - ( x * x * QQ / ( QQ + x * t ) * t / 4. / M2 ) * Etilde.Rho2() );
    // c_dvcs_unp(Feff,Feff*)
    c_dvcs_effeffs = f * f * c_dvcs_ffs;
    // c_dvcs_unp(Feff,F*)
    c_dvcs_efffs = f * c_dvcs_ffs;
    // dvcs c_n coefficients (BKM10 eqs. [2.18], [2.19])
    c0_dvcs_10 = 2. * ( 2. - 2. * y + y * y  + ee / 2. * y * y ) / ( 1. + ee ) * c_dvcs_ffs + 16. * K * K / pow(( 2. - x ), 2) / ( 1. + ee ) * c_dvcs_effeffs;
    c1_dvcs_10 = 8. * K / ( 2. - x ) / ( 1. + ee ) * ( 2. - y ) * c_dvcs_efffs;
    
    return DVCS10_c1fit = Gamma * GeV2nb * c1_dvcs_10 * 1. / ( y * y * QQ );
}


Double_t TBKM::I_UU_10(Double_t *kine, Double_t phi, Double_t F1, Double_t F2, TComplex *t2cffs, TString twist = "t2") { // Interference Unpolarized Cross Section (Liuti's style)

    // Get BH propagators and set the kinematics
    BHLeptonPropagators(kine, phi);

    // Set the CFFs
    SetCFFs(t2cffs); // Etilde CFF does not appear in the interference

    // Get A_UU_I, B_UU_I and C_UU_I interference coefficients
    ABC_UU_I_10(kine, phi, A_U_I, B_U_I, C_U_I, twist);

    // BH-DVCS interference squared amplitude
    I_10 = 1. / ( x * y * y * y * t * P1 * P2 ) * ( A_U_I * ( F1 * H.Re() - t / 4. / M2 * F2 * E.Re() ) + B_U_I * ( F1 + F2 ) * ( H.Re() + E.Re() ) + C_U_I * ( F1 + F2 ) * Htilde.Re() );

    I_10 = GeV2nb * I_10; // convertion to nb

    return dsigma_I_10 = Gamma * I_10;
}
//_______________________________________________________________________________________________________________________________
void TBKM::ABC_UU_I_10(Double_t *kine, Double_t phi, Double_t &A_U_I, Double_t &B_U_I, Double_t &C_U_I, TString twist = "t2") { // Get A_UU_I, B_UU_I and C_UU_I interference coefficients (BKM10)

    SetKinematics(kine);

    // F_eff = f * F
    if(twist == "t2") f = 0; // F_eff = 0 ( pure twist 2)
    if(twist == "t3") f = - 2. * xi / ( 1. + xi );
    if(twist == "t3ww") f = 2. / ( 1. + xi );

    // Interference coefficients  (BKM10 Appendix A.1)
    // n = 0 -----------------------------------------
    // helicity - conserving (F)
    C_110 = - 4. * ( 2. - y ) * ( 1. + sqrt( 1 + ee ) ) / pow(( 1. + ee ), 2) * ( Ktilde_10 * Ktilde_10 * ( 2. - y ) * ( 2. - y ) / QQ / sqrt( 1 + ee )
            + t / QQ * ( 1. - y - ee / 4. * y * y ) * ( 2. - x ) * ( 1. + ( 2. * x * ( 2. - x + ( sqrt( 1. + ee ) - 1. ) / 2. + ee / 2. / x ) * t / QQ + ee ) / ( 2. - x ) / ( 1. + sqrt( 1. + ee ) ) ) );
    C_110_V = 8. * ( 2. - y ) / pow(( 1. + ee ), 2) * x * t / QQ * ( ( 2. - y ) * ( 2. - y ) / sqrt( 1. + ee ) * Ktilde_10 * Ktilde_10 / QQ
              + ( 1. - y - ee / 4. * y * y ) * ( 1. + sqrt( 1. + ee ) ) / 2. * ( 1. + t / QQ ) * ( 1. + ( sqrt ( 1. + ee ) - 1. + 2. * x ) / ( 1. + sqrt( 1. + ee ) ) * t / QQ ) );
    C_110_A = 8. * ( 2. - y ) / pow(( 1. + ee ), 2) * t / QQ * ( ( 2. - y ) * ( 2. - y ) / sqrt( 1. + ee ) * Ktilde_10 * Ktilde_10 / QQ * ( 1. + sqrt( 1. + ee ) - 2. * x ) / 2.
              + ( 1. - y - ee / 4. * y * y ) * ( ( 1. + sqrt( 1. + ee ) ) / 2. * ( 1. + sqrt( 1. + ee ) - x + ( sqrt( 1. + ee ) - 1. + x * ( 3. + sqrt( 1. + ee ) - 2. * x ) / ( 1. + sqrt( 1. + ee ) ) )
              * t / QQ ) - 2. * Ktilde_10 * Ktilde_10 / QQ ) );
    // helicity - changing (F_eff)
    C_010 = 12. * sqrt(2.) * K * ( 2. - y ) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * ( ee + ( 2. - 6. * x - ee ) / 3. * t / QQ );
    C_010_V = 24. * sqrt(2.) * K * ( 2. - y ) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * x * t / QQ * ( 1. - ( 1. - 2. * x ) * t / QQ );
    C_010_A = 4. * sqrt(2.) * K * ( 2. - y ) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * t / QQ * ( 8. - 6. * x + 5. * ee ) * ( 1. - t / QQ * ( ( 2. - 12 * x * ( 1. - x ) - ee )
              / ( 8. - 6. * x + 5. * ee ) ) );
    // n = 1 -----------------------------------------
    // helicity - conserving (F)
    C_111 = -16. * K * ( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * ( ( 1. + ( 1. - x ) * ( sqrt( 1 + ee ) - 1. ) / 2. / x + ee / 4. / x ) * x * t / QQ - 3. * ee / 4. )
            - 4. * K * ( 2. - 2. * y + y * y + ee / 2. * y * y ) * ( 1. + sqrt( 1 + ee ) - ee ) / pow(sqrt( 1. + ee ), 5) * ( 1. - ( 1. - 3. * x ) * t / QQ
            + ( 1. - sqrt( 1 + ee ) + 3. * ee ) / ( 1. + sqrt( 1 + ee ) - ee ) * x * t / QQ ) ;
    C_111_V = 16. * K / pow(sqrt( 1. + ee ), 5) * x * t / QQ * ( ( 2. - y ) * ( 2. - y ) * ( 1. - ( 1. - 2. * x ) * t / QQ ) + ( 1. - y - ee / 4. * y * y )
              * ( 1. + sqrt( 1. + ee ) - 2. * x ) / 2. * ( t - tmin ) / QQ );
    C_111_A = -16. * K / pow(( 1. + ee ), 2) * t / QQ * ( ( 1. - y - ee / 4. * y * y ) * ( 1. - ( 1. - 2. * x ) * t / QQ + ( 4. * x * ( 1. - x ) + ee ) / 4. / sqrt( 1. + ee ) * ( t - tmin ) / QQ )
              - pow(( 2. - y ), 2) * ( 1. - x / 2. + ( 1. + sqrt( 1. + ee ) - 2. * x ) / 4. * ( 1. - t / QQ ) + ( 4. * x * ( 1. - x ) + ee ) / 2. / sqrt( 1. + ee ) * ( t - tmin ) / QQ ) );
    // helicity - changing (F_eff)
    C_011 = 8. * sqrt(2.) * sqrt( 1. - y - ee / 4. * y * y ) / pow(( 1. + ee ), 2) * ( pow(( 2. - y ), 2) * ( t - tmin ) / QQ * ( 1. - x + ( ( 1. - x ) * x + ee / 4. ) / sqrt( 1. + ee ) * ( t - tmin ) / QQ )
            + ( 1. - y - ee / 4. * y * y ) / sqrt( 1 + ee ) * ( 1. - ( 1. - 2. * x ) * t / QQ ) * ( ee - 2. * ( 1. + ee / 2. / x ) * x * t / QQ ) );
    C_011_V = 16. * sqrt(2.) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * x * t / QQ * ( pow( Ktilde_10 * ( 2. - y ), 2) / QQ + pow(( 1. - ( 1. - 2. * x ) * t / QQ ), 2) * ( 1. - y - ee / 4. * y * y ) );
    C_011_A = 8. * sqrt(2.) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * t / QQ * ( pow( Ktilde_10 * ( 2. - y ), 2) * ( 1. - 2. * x ) / QQ + ( 1. - ( 1. - 2. * x ) * t / QQ )
              * ( 1. - y - ee / 4. * y * y ) * ( 4. - 2. * x + 3. * ee + t / QQ * ( 4. * x * ( 1. - x ) + ee ) ) );
    // n = 2 -----------------------------------------
    // helicity - conserving (F)
    C_112 = 8. * ( 2. - y ) * ( 1. - y - ee / 4. * y * y ) / pow(( 1. + ee ), 2) * ( 2. * ee / sqrt( 1. + ee ) / ( 1. + sqrt( 1. + ee ) ) * pow(Ktilde_10, 2) / QQ + x * t * ( t - tmin ) / QQ / QQ
            * ( 1. - x - ( sqrt( 1. + ee ) - 1. ) / 2. + ee / 2. / x ) );
    C_112_V = 8. * ( 2. - y ) * ( 1. - y - ee / 4. * y * y ) / pow(( 1. + ee ), 2) * x * t / QQ * ( 4. * pow(Ktilde_10, 2) / sqrt( 1. + ee ) / QQ + ( 1. + sqrt( 1. + ee ) - 2. * x ) / 2. * ( 1. + t / QQ ) * ( t - tmin ) / QQ );
    C_112_A = 4. * ( 2. - y ) * ( 1. - y - ee / 4. * y * y ) / pow(( 1. + ee ), 2) * t / QQ * ( 4. * ( 1. - 2. * x ) * pow(Ktilde_10, 2) / sqrt( 1. + ee ) / QQ - ( 3. -  sqrt( 1. + ee ) - 2. * x + ee / x ) * x * ( t - tmin ) / QQ );
    // helicity - changing (F_eff)
    C_012 = -8. * sqrt(2.) * K * ( 2. - y ) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * ( 1. + ee / 2. ) * ( 1. + ( 1. + ee / 2. / x ) / ( 1. + ee / 2. ) * x * t / QQ );
    C_012_V = 8. * sqrt(2.) * K * ( 2. - y ) * sqrt( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * x * t / QQ * ( 1. - ( 1. - 2. * x ) * t / QQ );
    C_012_A = 8. * sqrt(2.) * K * ( 2. - y ) * sqrt( 1. - y - ee / 4. * y * y ) / pow(( 1. + ee ), 2) * t / QQ * ( 1. - x + ( t - tmin ) / 2. / QQ * ( 4. * x * ( 1. - x ) + ee ) / sqrt( 1. + ee ) );
    // n = 3 -----------------------------------------
    // helicity - conserving (F)
    C_113 = -8. * K * ( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * ( sqrt( 1. + ee ) - 1. ) * ( ( 1. - x ) * t / QQ + ( sqrt( 1. + ee ) - 1. ) / 2. * ( 1. + t / QQ ) );
    C_113_V = -8. * K * ( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * x * t / QQ * ( sqrt( 1. + ee ) - 1. + ( 1. + sqrt( 1. + ee ) - 2. * x ) * t / QQ );
    C_113_A = 16. * K * ( 1. - y - ee / 4. * y * y ) / pow(sqrt( 1. + ee ), 5) * t * ( t - tmin ) / QQ / QQ * ( x * ( 1. - x ) + ee / 4. );

    // A_U_I, B_U_I and C_U_I
    A_U_I = C_110 + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * C_010 + ( C_111 + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * C_011 ) * cos( PI - (phi * RAD) )
            + ( C_112 + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * C_012 ) * cos( 2. * ( PI - (phi * RAD) ) ) + C_113 * cos( 3. * ( PI - (phi * RAD) ) );
    B_U_I = xi / ( 1. + t / 2. / QQ ) * ( C_110_V + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * C_010_V + ( C_111_V + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * C_011_V ) * cos( PI - (phi * RAD) )
            + ( C_112_V + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * C_012_V ) * cos( 2. * ( PI - (phi * RAD) ) ) + C_113_V * cos( 3. * ( PI - (phi * RAD) ) ) );
    C_U_I = xi / ( 1. + t / 2. / QQ ) * ( C_110 + C_110_A + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * ( C_010 + C_010_A ) + ( C_111 + C_111_A + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f
            * ( C_011 + C_011_A ) ) * cos( PI - (phi * RAD) ) + ( C_112 + C_112_A + sqrt(2) / ( 2. - x ) * Ktilde_10 / sqrt(QQ) * f * ( C_012 + C_012_A ) ) * cos( 2. * ( PI - (phi * RAD) ) )
            + ( C_113 + C_113_A ) * cos( 3. * ( PI - (phi * RAD) ) ) );
}

