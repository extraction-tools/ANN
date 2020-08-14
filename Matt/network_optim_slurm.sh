#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH -p standard
#SBATCH -A spinquest
#SBATCH --job-name=network_batch_optim.py

#SBATCH --output=network_optim%A_%a.out
#SBATCH --error=network_optim_%A_%a.error

python network_batch_optim.py ${SLURM_ARRAY_TASK_ID}
