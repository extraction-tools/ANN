#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH --mail-type=END
#SBATCH --mail-user=mohit@virginia.edu
singularity run --nv /home/mvs2dq/tensorflow-2.1.0-py37.sif localfit_v2.py
