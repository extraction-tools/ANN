## Generated pseudodata on 11/19/2020

### *Changes in class TBHDVCS:*

* *jcob* parameter modified.
* In *GetBHUUxs* an extra factor 2 was removed.
* In *GetIUUxs* the AUUI, BUUI and CUUI coefficients were modified to the last version of the paper [arXiv:1903.05742 [hep-ph]](https://arxiv.org/abs/1903.05742).
* Functions with polarized configurations that are not being used were removed.
 
### *Changes in class TFormFactors:*

* A new parametrization of F1 and F2 is included (Kelly's parametrization). 

Note: This new parametrization is the one used to generate the data. To obtain the values of F1 and F2 use the functions `Double_t TFormFactors::ffF1_K(Double_t t)` and `Double_t TFormFactors::ffF2_K(Double_t t)`.

### Pseudodata

Cross section values are generated in the file **genytree.C** without selecting the CFFs reproducibility within 20%.

After a least Squares fit was able to reproduce the *true* values within 20% difference, as seen in the file **CFFs_SelKine_0.05.pdf**, there remain a total of 19 kinematic settings.

The generated cross section F at the selected kinematics was obtained with a 5% variance.

The pseudodata output values are shown on the file **dvcs_xs_11-19-20_0.05.csv**.
