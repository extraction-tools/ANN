
# Neural Network Architectures for Predicting Compton Form Factors

This is the repo the code I'm developing for UVA's spin physics research group. The group is using artificial neural networks to predict the 3 Compton form factors $\textrm{Re}(\mathcal{H}), \textrm{Re}(\mathcal{E}), \textrm{and}$ $\textrm{Re}(\mathcal{\tilde{H}})$. These form factors are then used generate an approximation curve of a function $F$. The neural networks are feed 4 variables that the Compton form factors depend on $x_B$, $Q^2$, $t$ and $k$.


# Files

The two folder BKM02 and BKM10 contain the code for the local fits using the BKM 2002 and BKM 2010 formalisms respectively.  The folders contain 2 folders: one with the code for the current baseline neural network architecture and one with the code for the radial basis function neural network I implemented based off the work *Neural  Networks and  Deep Learning* by Charu C. Aggarwal. These neural networks are especially suited for function approximation.


# Radial Basis Function (RBF) Neural Networks

In their simplest form RBF networks consist of an input layer, a hidden layer and an output layer. Where they differ from regular neural networks is that each of the $m$ neurons in the hidden layer uses a radial basis function as an activation function. The most common type of RBF function used is a Gaussian function. 
So for the $i$th neuron in the hidden layer its activation function is:

$$\Phi_i(\bar{X})=\textrm{exp}(-\frac{|| \bar{X} - \bar{\mu_i}||^2}{2\sigma_i^2}) \quad \forall i \in \{1,...,m\}$$

The parameters $\bar{\mu}_i$ and $\sigma_{i}$ are the prototype vector and bandwidth of the $i$th neuron respectively. 
The prototype vectors are learned from the training data in a unsupervised way either by randomly selecting $m$ points from the training data or via clustering. 

In the code in this repo I use Scikit-learn's K-means clustering algorithm to learn the prototype vectors. The bandwidth's are all set to the same value and the code provides the option to initialize them all either to $2d_{avg}$ or $\frac{d_{max}}{m}$. 

Where $d_{avg}$ and $d_{max}$ are the average distance between prototype vectors and the maximum distance between prototype vectors respectively. In my implementation of the RBF network I also follow each RBF layer with a linear layer.
