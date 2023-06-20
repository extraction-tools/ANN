#!/bin/bash

dir_macros=$(dirname $(readlink -f $BASH_SOURCE))

jobname=$1
njobs=$2

echo "njobs=$njobs"

for (( id=1; id<=$[$njobs]; id++ ))
do  
  echo "submitting job number = $id"
  sbatch grid.slurm $id $jobname
  
done