#!/usr/bin/env bash
#SBATCH -p standard
#SBATCH --output=result_%a.out
#SBATCH --error=result_%a.err
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -A spinquest

module purge
module load gcc/9.2.0 lhapdf
echo $LHAPDF_DATA_PATH
ls $LHAPDF_DATA_PATH
python -c "import lhapdf"
python -V
module load gcc/9.2.0 lhapdf
source activate tf-2.7
export LD_LIBRARY_PATH=~/.conda/envs/tf-2.7/lib:$LD_LIBRARY_PATH

python Create_Sample_Data.py $1