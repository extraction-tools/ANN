VERSION_0: 5/24/2021

The files that you need to run the localfit for the second pseudo data from liliet:
1. localfit.py
2. BHDVCS_torch.py
3. Lorentz_Vector.py
4. job.slurm

Run using sbatch job.slurm. It will produces 200 Replica (max number). In order to run more replica, redo "sbatch job.slurm and change the ouput end error files name (*.out *.err)

In the localfit.py specifiy the number of sets you want to analyze in line 39

copy the pytorch sif file to your home directory before you run the job:
copy $CONTAINERDIR/pytorch-1.5.1.sif <your home directory> 

Don't forget modify job.slurm to your directory

