## Generated pseudodata on May 2021

### *Theoretical Formulation:*

* The Total DVCS cross section distributions (F) was model with the VA Formulation [arXiv:1903.05742v3 [hep-ph]](https://arxiv.org/abs/1903.05742v3).

*  The total cross section (F) is calculated by adding the BH (`Double_t TVA1_UU::GetBHUU(Double_t phi, Double_t F1, Double_t F2)`), pure DVCS (constant values included on the data file) and the BH-DVCS (`Double_t TVA1_UU::GetIUU(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde)`) cros sections.

* The Form Factors F1 and F2 are calculated using Kelly's parametrization. To obtain the values of F1 and F2 use the functions `Double_t TFormFactors::ffF1_K(Double_t t)` and `Double_t TFormFactors::ffF2_K(Double_t t)`.

