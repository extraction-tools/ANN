#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard

module purge
module load anaconda/2020.11-py3.8
module load singularity/3.7.1
module load tensorflow/2.1.0-py37

singularity run --nv $CONTAINERDIR/tensorflow-2.7.0.sif localfit_v2.py
