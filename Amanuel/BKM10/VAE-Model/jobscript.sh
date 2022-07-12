#!/bin/bash

dir_macros=$(dirname $(readlink -f $BASH_SOURCE))

jobname=$1
njobs=$2

echo "njobs=$njobs"
echo "nevents=$nevents"

work=/scratch/$USER/ANN/$jobname


mkdir -p $work
chmod -R 01755 $work

cd $dir_macros


for (( id=1; id<=$njobs; id++ ))
do  
  mkdir -p $work/$id/
  chmod -R 01755 $work/$id
  cd $work/$id/
  cp $dir_macros/*.py .
  cp $dir_macros/*.txt .
  cp $dir_macros/*.csv .
  cp $dir_macros/*.slurm .
  
  echo "submitting job number = $id"
  sbatch grid.slurm
    

done 
