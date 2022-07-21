#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=rbf.out
#SBATCH --error=rbf.err
#SBATCH --partition=standard
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

time singularity run --nv ~/pytorch-1.8.1.sif ann-rbf.py
