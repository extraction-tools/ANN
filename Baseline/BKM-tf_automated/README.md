# Directory Contents

This directory contains the following files and folders:

1. **modified_fits**: This folder contains various modified local fit files.
   - The modified fit files are located in subfolders named after the changed local fit names.
   - The CFF_Data for each modified fit is located in the corresponding subfolder: `modified_fits/_name_of_fit/CFF_Data`.

2. **comparison.py**: This script reads through the CFF_Data of a specific modified fit and updates the relevant information in `results.csv`.

3. **fit.py**: This file contains the updated local fit code for the BKM method. It is capable of producing validation loss and epoch loss.

4. **grid.slurm**: This file defines the system requirements for running parallelized batch jobs on the Rivanna cluster.

5. **jobscript.sh**: This script automates the entire process.
   - Running `jobscript.sh` executes the entire workflow.
   - It starts by modifying `fit.py` based on `ANN.csv` using `modify_fit.py`.
   - It then submits multiple jobs using the modified fit files and `grid.slurm`.
   - After the jobs finish, it processes the output using `comparison.py`.
   - Finally, it updates the values in `ANN.csv` based on the results.

6. **modify_fit.py**: This script takes information from `results.csv` and creates new modified fit files. Each modified fit file is named in the format "number_layers_number_nodes_activation_function_fit.py".

7. **pseudoKM15_New_FormFactor.csv**: This file contains the updated pseudo-data values.

8. **results.csv**: This Excel spreadsheet is used to update the tests that need to be implemented.
   - After running the jobs, it contains the results of the tests.

# Execution

To automate the entire process, execute the following command:

```bash
./jobscript.sh
