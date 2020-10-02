# Error Propagation to the CFFs

### Method 1:
Generate F distribution as a function of phi with all the CFFs standard deviation set to 10% at the same time.

*Files*
  - **FvarVsPhi.pdf** Standard deviation of F obtained as a function of phi when the values of all the CFFs are generated with a 10% standard deviation. 

### Method 2:

Use 5% variance in F and look at the resulting standard deviation in each of the CFFs one at a time, fixing the other two.

*Files*

- **ErrorPropagationToCFFs.pdf** Presentation slides containing the method steps and summary or results.
- **ReH_var.pdf** Obtained standard deviation of ReH for all the sets.
- **ReE_var.pdf** Obtained standard deviation of ReE for all the sets.
- **ReHtilde_var.pdf** Obtained standard deviation of ReHtilde for all the sets.
- **Deviation.pdf** Deviation of the CFFs from the values used to generate F.
- **cffs_error_prop.csv** Print out of the obtained values of the CFFs standard deviation, the columns are:
  * set, point
  * kinematics (k, QQ, x_b, t)
  * phi
  * F: Mean of the generated F values with 5% variation at fixed values of the CFFs that had no errors.
  * F_var(%): Variation of F that was fixed at 5% for all phi.
  * ReH_true: Set value of ReH used to generate the F distribution.
  * ReH_var(%): Standard deviation in % of ReH obtained for the generated F values keeping ReE and ReHtilde fixed to ReE_true, ReHtilde_true with one errors.
  * Analogously to ReH_true and ReH_var, the corresponding values of ReE and ReHtilde are shown in the following columns.
