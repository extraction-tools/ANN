#!/bin/bash

# Step 1: Modify fit.py based on ANN.csv
# modify_fit.py produces n number of modified_fit.py files depending on changes. new file name will be modified_fit_(case change).py
python modify_fit.py ANN.csv

# Set the desired number of parallel runs
parallel_runs=100

# Step 2: Submitting Multiple Jobs
for modified_fit in modified_fits/*.py; do
    job_name=$(basename "$modified_fit" .py)
    
    for ((i=1; i<=parallel_runs; i++)); do
        sbatch --job-name="fit_job_${job_name}_${i}" --output="fit_output_${job_name}_${i}.txt" grid.slurm "$modified_fit"
    done
done

# Step 4: Wait for Jobs to Finish
echo "Waiting for jobs to finish..."
while [[ $(squeue -u your_username | wc -l) -gt 1 ]]; do
    sleep 10s
done

# Step 5: Process Output with compare.py
# Searches through modified_fit.py files within modified_fits folder
for modified_fit_folder in modified_fits/*; do
    if [[ -d "$modified_fit_folder" ]]; then
        folder_name=$(basename "$modified_fit_folder")
        # compare.py will update values to ANN.csv
        python compare.py --folder_path "$modified_fit_folder" --output "result_${folder_name}.txt"
    fi
done
