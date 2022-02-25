# Sammy's Hyperparameter-Tuning Demo of Method 2 with Rivanna

# Introduction
My current task is the implementation of a full hyperparameter search using Rivanna. In this README, I detail my Rivanna setup, techniques, and further tasks.

# Rivanna Setup Walkthrough Instructions
1. First, set up the UVA VPN according to the following link:(https://virginia.service-now.com/its/?id=itsweb_kb_article&sys_id=f24e5cdfdb3acb804f32fb671d9619d0). You won't be able to remote into Rivanna without having this VPN turned on and connected. (connecting with UVA from anywhere works fine)

2. Make sure you have a terminal environment on your machine to access Rivanna (Windows powershell, bash Windows subsystem, Mac terminal all work well for this case). Connect your VPN with the UVA from anywhere option, and have it on before you SSH.

3. Remote into Rivanna with the following command in your terminal (NOTE: Prof Keller must have added your account to have access to this). Use the following command: ssh -Y your_computing_id@rivanna.hpc.virginia.edu. This should prompt your netbadge password if you successfully connected.

4. Once inside you have to copy over files from your remote machine to your directory in Rivanna. This can be done with an Secure Copy Linux command (SCP) in your local terminal as follows: Open a separate terminal and cd into the directory that contains the required source code. Type in the command: SCP filename your_computing_id@rivanna.hpc.virginia.edu:PATH , where PATH can be found by typing the pwd command in the terminal with Rivanna. Further, these required files can be found directly in my Github directory under the name, Rivannas.

4. Run a job within the Rivannas directory with the following command: "sbatch --array=0-14 Job1.slurm". This will produce 15 different jobs queued in UVA's Rivanna Job Manager and can be viewed with the following link: https://rivanna-portal.hpc.virginia.edu/pun/sys/activejobs/?jobcluster=all&jobfilter=user

# Rivanna Test Run Walkthrough

# Current Hyperparameter 
Learning Rate: interval between (0.1,0.0000001) <br />
Activation Functions: ["tanh", "relu", "selu", "elu"] <br />
Number of Nodes: [4,8,20,16,32,64,128] <br />
Number of hidden layers: 1-5 <br />
Number of dropout layers: 1-3 <br />
Optimizers: ["Adam", "SGD"] <br />

# Hyperparameters that should be considered, however haven't been implemented yet:
Dropout layer placement with respect to hidden layers: TBD <br />
Nodes per layer (shape): TBD <br />


## Remainning Tasks:
 1. Modify the code to allow for hyperparameter-tuning by adjusting the number of hidden layers
 2. Modify the code to allow for hyperparameter-tuning by adjusting the number of dropout layers
 3. Modify the code to allow for hyperparameter-tuning by adjusting the number of dropout layers
 4. Modify the code to allow for hyperparameter-tuning by adjusting the number of hidden layers and nodes per dropout layer (changing shape in a grid search fashion)
 5. Implement the consistent global-fitting schema provided by Nathan
 6. Add in functionality that stores searched hyperparameter results according the the schema
 7. Have a functionality that prevents duplicate 


# Sammy's Custom Optimizer and Custom Loss Function Template Code Creation (Date: 2/25/22)

# Introduction
My current task is the implementation of custom loss functions and optimizers within the current method. The purpose is to allow for future custom designs in loss functions <br />
that aren't already included in common ML-language libraries (i.e. the stigmergic loss function). <br />

Resources used:
- Custom Optimizer: https://cloudxlab.com/blog/writing-custom-optimizer-in-tensorflow-and-keras/ 

# Plan of action
1. Create custom loss function template code compatible with all of the current methods.
2. Baseline the loss function against common libraries to ensure working implementation (Mean-squared error loss).
3. Create custom optimizer template code compatible with all of the current methods.
4. Baseline the custom optimizers against common libraries to ensure working implementation (Stochastic Gradient Descent, Adam Optimizer).
5. Explore custom loss function and optimizer designs that may yield different performances within the existing models. 

# Custom Loss Function Implementation 
- New custom loss function template added and pushed for entire group's use

# Baselining 
- Successfully baselined against MSE loss function

# Custom Optimizer Implementation 
- New custom loss function template added and pushed for entire group's use

# Baselining 
- Baselining still in-progree against the MSE loss function 


# Remainning tasks
- Design a practical custom loss function to potentially yield different model performance
- Design a practical custom optimizer to potentially yield different model performance
- Implement Stimergic loss





