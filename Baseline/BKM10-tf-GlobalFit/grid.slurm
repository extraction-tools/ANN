#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:01:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=dev
#SBATCH -A spinquest
#SBATCH --array=0-100

module purge
module load anaconda/2020.11-py3.8
module load singularity/3.7.1
module load tensorflow

singularity run --nv /home/$USER/tensorflow-2.7.0.sif localfit_v2.py $SLURM_ARRAY_TASK_ID
