#!/bin/bash

dir_macros=$(dirname $(readlink -f $BASH_SOURCE))

jobname=$1
njobs=$2

echo "njobs=$njobs"

work=/project/ptgroup/Devin/Neural_Network/Sample_Data/$jobname

mkdir -p $work
chmod -R 01755 $work

# cd $dir_macros

for (( id=1; id<=$[$njobs]; id++ ))
do  
  echo "submitting job number = $id"
  sbatch Create_Sample_Data.slurm $id
  
done