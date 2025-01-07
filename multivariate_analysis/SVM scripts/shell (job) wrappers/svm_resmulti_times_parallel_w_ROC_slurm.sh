#!/bin/bash
#SBATCH --mem=50G
#SBATCH --array=1-100
#SBATCH --time=1-00:00:00

python3 /cbica/projects/ash_pfn_sex_diff_abcd/dropbox/run_2fold_svm_resmulti_times_parallel_w_ROC.py -5 10 $SLURM_ARRAY_TASK_ID
