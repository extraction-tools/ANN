#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=sammyTest_%a.out
#SBATCH -c 1
#SBATCH -A spinquest

module purge
module load anaconda/2019.10-py3.7
#module load singularity/3.5.2
module load singularity
module load tensorflow/2.1.0-py37

#python /home/sl8rn/Rivannas/General2.py ${SLURM_ARRAY_TASK_ID}
singularity run --nv /home/$USER/tensorflow-2.1.0-py37.sif /home/sl8rn/Rivannas/General2.py ${SLURM_ARRAY_TASK_ID}
