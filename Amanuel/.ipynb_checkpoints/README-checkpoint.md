# Neural Network Architectures for Predicting Compton Form Factors

This is the repo the code I'm developing for UVA's spin physics research group. The group is using artificial neural networks to predict the 3 Compton form factors $\textrm{Re}(\mathcal{H})$, $\textrm{Re}(\mathcal{E})$, and $\textrm{Re}(\mathcal{\tilde{H}})$. These form factors are then used generate an approximation curve of a function $F$. The neural networks are feed 4 variables that the Compton form factors depend on $x_B$, $Q^2$, $t$ and $k$.


# Files

The two folder BKM02 and BKM10 contain the code for the local fits using the BKM 2002 and BKM 2010 formulations respectively.  The folders contain 2 folders: one with the code for the current baseline neural network architecture and one with the code for the radial basis function neural network I implemented based off the work *Neural  Networks and  Deep Learning* by Charu C. Aggarwal.


# Radial Basis Function (RBF) Neural Networks