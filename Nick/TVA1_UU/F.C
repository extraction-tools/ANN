#include "TVA1_UU.h"
#include "TVA1_UU.cxx"

class F {
    private:
        TVA1_UU *bhdvcs;

    public:
        F(){bhdvcs = new TVA1_UU;}
        ~F(){delete bhdvcs;}
        Double_t TotalUUXS(Double_t _phi, Double_t _k, Double_t _QQ,  Double_t _xB,  Double_t _t, Double_t _F1,
                           Double_t _F2,  Double_t _ReH,  Double_t _ReE,  Double_t _ReHtilde, Double_t _dvcs);
};


Double_t F::TotalUUXS(Double_t _phi, Double_t _k, Double_t _QQ,  Double_t _xB,  Double_t _t, Double_t _F1,
                      Double_t _F2,  Double_t _ReH,  Double_t _ReE,  Double_t _ReHtilde, Double_t _dvcs){

    Double_t AUUI, BUUI, CUUI;

    // Set QQ, xB, t and k
    bhdvcs->SetKinematics( _QQ, _xB, _t, _k );

    Double_t xsbhuu = bhdvcs->GetBHUU(_phi, _F1, _F2); // BH cross section
    Double_t xsiuu = bhdvcs->GetIUU(_phi, _F1, _F2, _ReH, _ReE, _ReHtilde);

    Double_t tot_sigma_uu = xsbhuu + xsiuu + _dvcs; // Constant added to account for DVCS contribution

    return tot_sigma_uu;
}