#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest
#SBATCH --array=0-5

module purge
module load gcc/9.2.0
module load lhapdf/6.3.0
module load anaconda/2020.11-py3.8
module load singularity/3.7.1
module load tensorflow

python3 NN_SIDIS_ModGen.py $SLURM_ARRAY_TASK_ID