#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH -c 1
#SBATCH -t 16:30:00
#SBATCH -A spinquest_standard
#SBATCH --array=0-10

module purge
module load anaconda/2020.11-py3.8
module load singularity/3.7.1
module load tensorflow/2.8.0

singularity run --nv /home/mag4ka/tensorflow-2.8.0.sif /home/mag4ka/Aaryan/Rivanna/Multi_err/Full_ML_fit_evaluation_Set2.py ${SLURM_ARRAY_TASK_ID}
