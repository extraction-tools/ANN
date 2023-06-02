1. Create Sample Data using the jobscript.sh file. Format: './jobscript.sh test *number of jobs*'
2. Make sure to specify location of where data is created
3. Using Merge.py to merge all of the csv files in that directory into one csv file
4. Using the merged csv file, import that to the Train_NN.py script, then tune architecture as needed. Summary and loss values will be logged to a csv file. 
5. Once finished, use Predict.py and Predict.slurm with another newly create sample dataset to predict values. Results will save to a csv file
6. Import previous csv file into Histogram.py and run in command line. A histogram of the differences of P - P_true will be generated. 
