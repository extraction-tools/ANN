#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
singularity run --nv /home/$USER/pytorch-1.8.1.sif localfit-rbf.py
