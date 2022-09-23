#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A polarizedtarget
singularity run --nv ~/pytorch-1.5.1.sif semiglobal.py 
