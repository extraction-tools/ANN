#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -A spinquest_standard

module purge
module load anaconda
python -V
source activate tf-2.7
conda install scikit-learn
export LD_LIBRARY_PATH=~/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH

python Train_NN_v3.py