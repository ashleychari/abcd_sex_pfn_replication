# ABCD Dataset PFN Sex Differences Replication Project

## Directories of this repository
- discovery and replication sample setup scripts: Contains all python and R scripts related to the creation of samples utilized for the univariate and multivariate analysis 
- Univariate analysis: Contains all python and R scripts related to the unviariate analysis of data with the use of GAMs
- Multivariate analysis: Contains all python and R scripts related to the multivariate analysis with the use of Support Vector Machines

## Discovery and replication sample setup scripts
- get_data_for_ridge.R: R script adapted from Arielle Keller, utilizes files from `FilesForAdam` directory to create discovery and replication samples 
- create_discovery_replication_samples.R: R script that removes siblings from both discovery and replication samples randomly

## Univariate analysis
### GAMs scripts
Contains all the scripts that run the GAMs analysis
- abcd_univariate_analysis: script that runs the GAMs analysis on the discovery dataset
- abcd_univariate_analysis_replication: script that runs the GAMs analysis on the replication dataset that was obtained from the discovery and replication sample setup scripts

### Barplot Scipts
Contains all of the scripts that go along with the creation of the stacked barplot figure.
- create_barplot.R: R script taken from Sheila to plot the number of vertices for each network
- make_barplot_mat.py: Create the matrix from the GAMs analysis zscores from each of the networks.

### workbench setup scripts
Contains all the of the python and R scripts to transform GAMs Z vectors into unthresholded absolute sum matrix and create cifti files for each of the networks.
- write_effect_map_abs_sum.py: Python script that gets unthresholded absolute sum for each of the GAMs Z vectors 
- write_effect_map_abs_sum_to_cifti: R script that creates cifti files that can be used to visualize unthresholded absolute sum matrix on brain

## Multivariate Analysis
### nonzero matrix creation scripts
- create_nonzero_matrix.py: Flattens 17 network per subject into one (1010004,) array, concatenates for all subjects, and gets nonzero columns to make final matrix
- create_network_specific_matrices.py: Python script that goes through and creates network specific matrices

### SVM scripts
- run_bootstrap_svm.py: Uses arielle's method with randomly sampling and uses discovery and replication as a fold
- run_svm.py: Uses arielle's method to run svm with discovery and replication each as a fold and identifies best C parameter
- run_network_specific_svm.py: 
- run_svm_2fold_cv.py: Runs 2 fold cross validation svm on discovery and replication sets separately, also identifies best C parameter, regresses out covariates (such as site, age, and motion). Seed is set so that replicable results can occur (meaning the same folds will be chosen whenever this script is ran)
- run_svm_2fold_resmulti_times_batched.py: Runs 2 fold cross validation svm for a permutation of times on discovery and replication sets separately, also identifies best C parameter, regresses out covariates. Does not set seed so that random shuffles can be created for the 2 fold cv at each iteration. Does all of this in batches. Used with `svm_resmulti_times_parallel_batched.sh` and `svm_submit_batched_jobs.py`.

### SVM weights barplot
- sum_abs_weights_matrix.py: Gets unthresholded absolute sum of the weights and creates matrix
- svm_weights_barplot.R: Script copied from Sheila's PNC PFN sex differences code to visualize stacked barplot of weight matrix

### SVM workbench visualization
- svm_workbench_visualization_matrix_replication.py: Python script to create coefficient weight matrix from the 2 fold cv svm runs for replication set
- svm_workbench_visualization_matrix.py: Python script to create coefficient weight matrix from the 2 fold cv svm runs for discovery set
- svm_workbench_visualization_matrix_multi_times_matrix.py: Python script intended to create coefficient weight matrix from the permutation 2 fold cv svm runs for discovery or replication set (script not tested yet)

### Haufe transformation scripts
- haufe_transform_weights.py - Script adapted from arielle and kevin that haufe transforms support vector machine coefficients


### Multivariate analysis steps
1. Run `create_nonzero_matrix.py` to setup the non-zero features matrix for subsequent steps. \\
2. Next, run `svm_submit_batched_jobs.py`, which will create 25 jobs that go through 4 iterations of the `run_svm_2fold_resmulti_times_batched.py` script. \\
3. Create the weight matrix by taking the mean of each of the coefficients from all the fold instances and save this weight matrix using the `svm_workbench_visualization_multi_times_matrix.py` script. \\
4. Use the `haufe_transform_weights.py` to do a haufe transformation on the coefficients 
5. Use this weight matrix from step 4 in the `svm_stacked_barplot.R` script to get the stacked barplot of feature importance for both Males and Females \\
6. Use the haufe transformed weight matrix in the `sum_abs_weights_matrix.py` to get the absolute value sum of the weights and save this array \\
7. Take the saved array and run the `abs_sum_weights_to_cifti.R` script to convert the matrix to a dscalar.nii file that can be visualized via workbench

# Location of files
All files are located in project folder `ash_pfn_sex_diff_abcd`.

## Dropbox folder
Contains:
1. data_for_ridge_030824.csv \\
2. discovery_sample_removed_siblings.csv \\
3. replication_sample_removed_siblings.csv \\
4. FilesForAdam/ \\
5. subject_data/ \\

## Results folder
Contains: 
1. Nonzero matrices for discovery and replication \\
2. Nonzero indicies for discovery and replication matrices \\
3. multivariate_analysis results \\
4. discovery univariate analysis results \\
5. replication univariate analysis results

