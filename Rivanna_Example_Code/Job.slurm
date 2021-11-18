#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH -c 1
#SBATCH -t 16:30:00
#SBATCH -A spinquest

module purge
module load anaconda/2020.11-py3.8
module load singularity/3.7.1
module load tensorflow/2.1.0-py37

singularity run --nv /home/$USER/tensorflow-2.1.0-py37.sif /home/cee9hc/ANN_GPD_Calc_Test/Full_ML_fit_evaluation_Set2.py ${SLURM_ARRAY_TASK_ID}
