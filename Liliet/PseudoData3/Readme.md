## Generated pseudodata (June 2021)

### *Theoretical Formulation:*

* The Total DVCS cross section distributions (F) was model with the [BKM(2002) Formulation](https://arxiv.org/abs/hep-ph/0112108).

*  The total cross section (F) is calculated by adding the BH (`Double_t TBKM02::BHUU(Double_t phi, Double_t F1, Double_t F2)`), pure DVCS (constant values included on the data file) and the BH-DVCS interference (`Double_t TBKM02::IUU(Double_t phi, Double_t F1, Double_t F2, TComplex t2cffs[4])`) cross sections.

* The Form Factors F1 and F2 are calculated using Kelly's parametrization. To obtain the values of F1 and F2 use the functions `Double_t TFormFactors::ffF1_K(Double_t t)` and `Double_t TFormFactors::ffF2_K(Double_t t)`.

### *Pseudodata Generation:*

* There are a total of 229 kinematic settings selected where a least squares fit was able to reproduce the *true* values of the CFFs within 15% difference.

* The cross section F at the selected kinematics was generated within a 5% variance.

* The are 45 points of F vs phi for each kinematic set.
