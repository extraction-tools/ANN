#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=07:00:00
#SBATCH --output=localFit.out
#SBATCH --error=localFit.err
#SBATCH --partition=standard
singularity run --nv ~/pytorch-1.8.1.sif ann-rbf.py
