# Notes

1) Please copy these files to your working folder
2) Refer to the ANNMethods file for block diagrams related to each method
3) There were 3 pseudo-data sets Produced by Liliet and those data files with details can be found on https://github.com/extraction-tools/ANN/tree/master/Liliet 
4) Each data file contains many "kinematic-settings" (check corresponding .csv files)
5) Cross-Section (so-called "F") is a function of:
* Kinematics: x_B, k, QQ, t, phi
* Other parameters: dvcs, F1, F2
* Compton Form Factors (CFFs)
7) **For a given "kinematic-setting"**: F is varying only as a function of phi (an angle), while all other parameters are being constant [including CFFs]
8) So the objective is to model these CFFs which satisfy for all kinematics to extract their behavior over the kinematic ranges and make predictions for new kinematic-setting(s). 
9) Step #7 is been called as "Local Fit", whereas Step #8 is being called "Global Fit" in those Methods.
