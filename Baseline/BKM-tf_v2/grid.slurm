#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest_standard

module purge
module load anaconda/2020.11-py3.8
module load singularity/3.7.1
module load singularity tensorflow/2.8.0

singularity run --nv $CONTAINERDIR/tensorflow-2.8.0.sif Fit_v2.py $1
