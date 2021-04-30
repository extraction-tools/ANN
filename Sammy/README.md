# Sammy's Hyperparameter-Tuning Demo of Method 2 with Rivanna

# Introduction
My current task is the implementation of a full hyperparameter search using Rivanna. In this README, I detail my Rivanna setup, techniques, and further tasks.

# Rivanna Setup Walkthrough Instructions




# Current Hyperparameter 
Learning Rate: interval between (0.1,0.0000001) <br />
Activation Functions: ["tanh", "relu", "selu", "elu"] <br />
Number of Nodes: [4,8,20,16,32,64,128] <br />
Number of hidden layers: 1-5 <br />
Number of dropout layers: 1-3 <br />
Optimizers: ["Adam", "SGD"] <br />

#Hyperparameters that should be considered, however haven't been implemented yet:
Dropout layer placement with respect to hidden layers: TBD
Nodes per layer (shape): TBD



## Remainning Tasks:
 1. Modify the code to allow for hyperparameter-tuning by adjusting the number of hidden layers
 2. Modify the code to allow for hyperparameter-tuning by adjusting the number of dropout layers
 3. Modify the code to allow for hyperparameter-tuning by adjusting the number of dropout layers
 4. Modify the code to allow for hyperparameter-tuning by adjusting the number of hidden layers and nodes per dropout layer (changing shape in a grid search fashion)
 5. Implement the consistent global-fitting schema provided by Nathan
 6. Add in functionality that stores searched hyperparameter results according the the schema
 7. Have a functionality that prevents duplicate 
