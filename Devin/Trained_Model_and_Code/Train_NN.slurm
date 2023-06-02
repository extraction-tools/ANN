#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 16
#SBATCH -t 48:00:00
#SBATCH -A spinquest

module purge
module load anaconda
module load gcc/9.2.0 lhapdf
echo $LHAPDF_DATA_PATH
ls $LHAPDF_DATA_PATH
python -c "import lhapdf"
python -V
source activate tf-2.7
conda install scikit-learn
export LD_LIBRARY_PATH=~/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH

python Train_NN.py