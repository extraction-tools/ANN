"""
My code for the RBF neural network
"""

import torch
import torch.nn as nn 
import numpy as np

"""
Code for the RBF Layer centers are the cluster centers of the data from 
sci-kit leants kmeans method. If useAvgDist is set to true than the bandwidth parameter (sigma)
of each neuron in the RBF layer is set to 2*average distance between clusters else it is set to
max distance between clusters / sqrt(number of clusters)
"""

class RBFLayer(nn.Module):
    def __init__(self, inFeatures, outFeatures, centers, useAvgDist):
        super(RBFLayer, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.centers = torch.from_numpy(centers).float()
        # calcualte distance of any two cluster centers in centers variable
        clusterDistances = [np.linalg.norm(c1 - c2) for c1 in centers for c2 in centers] 
        dMax = max(clusterDistances)
        dAvg = sum(clusterDistances)  /  len(clusterDistances)
        # tensor of sigmas of dimension 1 x len(centers) 
        if useAvgDist == True:
             self.sigma = torch.full( (1, len(self.centers)), (2*dAvg) )
        else:
            self.sigma = torch.full( (1, len(self.centers)), ( dMax / np.sqrt(len(self.centers)) ) )
        

    def forward(self, input):
        # Input has shape batchSize x inFeature
        batchSize = input.size(0)

        mu = self.centers.view(len(self.centers), -1).repeat(batchSize, 1, 1)
        X = input.view(batchSize, -1).unsqueeze(1).repeat(1, len(self.centers), 1)        
    
        #Phi = torch.exp( -(torch.pow(X-mu, 2).sum(2, keepdim=False) / (2*torch.pow(self.sigma, 2))) )
        Phi = torch.exp(-self.sigma.mul((X-mu).pow(2).sum(2, keepdim=False).sqrt() ) )
        #Phi /= (1e-9 + Phi.sum(dim=-1)).unsqueeze(-1)
        return(Phi)
        
        
class RBFNet(nn.Module):
    def __init__(self, inFeatures, outFeatures, centers, useAvgDist):
        super(RBFNet, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.rbf = RBFLayer(inFeatures, len(centers), centers, useAvgDist)
        self.linear = nn.Linear(len(centers), outFeatures)
    
    def forward(self, x):

        x = self.rbf(x)
        
        x = self.linear(x)
        # tanh here seems to smooth out curve, needs more testing
        # x = torch.tanh(x)
        return(x)
    
    
class RBFNet2Layer(nn.Module):
    def __init__(self, inFeatures, outFeatures, centers1, centers2, useAvgDist):
        super(RBFNet2Layer, self).__init__()
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.rbf1 = RBFLayer(inFeatures, len(centers1), centers1, useAvgDist)
        self.linear1 = nn.Linear(len(centers1), len(centers2))
        self.rbf2 = RBFLayer(len(centers2), len(centers2), centers2, useAvgDist)
        self.linear2 = nn.Linear(len(centers2), outFeatures)
    
    def forward(self, x):
        x = self.rbf1(x)
        x = self.linear1(x)
        x = self.rbf2(x)
        x = self.linear2(x)
        return(x.squeeze(1))