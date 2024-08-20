#!/bin/bash
#SBATCH --mem=50G
#SBATCH --array=1-1000
#SBATCH --time=1-00:00:00

python3 /cbica/projects/ash_pfn_sex_diff_abcd/dropbox/svm_1000_permutation_scrambled_outcome_parallel.py -5 10 $SLURM_ARRAY_TASK_ID
