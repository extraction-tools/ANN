#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=standard
#SBATCH -A spinquest_standard

module load gcc/11.2.0
module load openmpi/4.1.4

source activate tf-2.7

job_id=$SLURM_ARRAY_TASK_ID
timestamp=$(date +"%Y%m%d%H%M%S")  # Generate a timestamp

# Specify the output directory
output_dir="./trash"

# Modify the file name with job ID or timestamp
output_file="${output_dir}/result_${job_id}_${timestamp}.out"

echo
srun python3 localfit_v4.py $job_id $1