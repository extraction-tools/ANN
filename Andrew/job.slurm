#!/bin/sh
#SBATCH --time=02:30:00
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --partition=parallel

# Set the number of tasks in the following line:
#SBATCH --ntasks=200

for ((i=0;i<$SLURM_NTASKS;i++))
  do

    # ---------------------------------------------------------------------
    # Customize this section to meet your needs.
    #
    # In this example, we submit tasks to analyze many data files.
    # Each data file has a name like "run*.rz".  We want to analyze
    # runs 5001 through 5010 (10 runs) so we've set "ntasks" to 10,
    # above.
    # 
    ## firstrun="5001";
    ## ((runnumber=$firstrun+$i))
    # Define the variable "command" so that it does what you want:
    ## command="analyze run$runnumber.rz"
      
       command="singularity run --nv /home/za2hd/pytorch-1.5.1.sif localfit.py"
    # ---------------------------------------------------------------------

     echo "Submitting task $i: $command"
     srun --cpus-per-task=1 --cpu_bind=cores --exclusive --nodes=1 --ntasks=1 $command 1> p5-slurm-$i.out 2> p5-slurm-$i.err &
done
wait
