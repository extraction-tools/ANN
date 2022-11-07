## Run the following commands on your terminal ###
module load anaconda/2020.11-py3.8 
module load singularity/3.7.1
module load tensorflow/2.8.0
cp $CONTAINERDIR/tensorflow-2.8.0.sif /home/$USER

##########################################################################
### change the path with the correct (where you are running the code) ####
### 1) Job.slurm                                                      #### 
### 2) Full_ML_fit_evaluation_Set2.py file                            ####
##########################################################################


### Then run the following command to submit the job ######
### here you can include the range of the kinematics #####

sbatch --array=0-2 Job.slurm
