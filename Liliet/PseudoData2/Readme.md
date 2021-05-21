## Generated pseudodata on May 2021

### *Theoretical Formulation:*

* The Total DVCS cross section distributions (F) was model with the VA Formulation [arXiv:1903.05742v3 [hep-ph]](https://arxiv.org/abs/1903.05742v3).

*  The total cross section (F) is calculated by adding the BH (`Double_t TVA1_UU::GetBHUU(Double_t phi, Double_t F1, Double_t F2)`), pure DVCS (constant values included on the data file) and the BH-DVCS (`Double_t TVA1_UU::GetIUU(Double_t phi, Double_t F1, Double_t F2, Double_t ReH, Double_t ReE, Double_t ReHtilde)`) cross sections.

* The Form Factors F1 and F2 are calculated using Kelly's parametrization. To obtain the values of F1 and F2 use the functions `Double_t TFormFactors::ffF1_K(Double_t t)` and `Double_t TFormFactors::ffF2_K(Double_t t)`.

### *Pseudodata Generation:*

* There are a total of 342 kinematic settings selected where a least squares fit was able to reproduce the *true* values of the CFFs within 20% difference.

* The cross section F at the selected kinematics was generated within a 5% variance.

* The are 45 points of F vs phi for each kinematic set.
