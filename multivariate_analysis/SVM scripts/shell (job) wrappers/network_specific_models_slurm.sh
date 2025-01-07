#!/bin/bash
#SBATCH --mem=50G
#SBATCH --array=1-100
#SBATCH --propagate=NONE
#SBATCH --time=1-00:00:00

python3 /cbica/projects/ash_pfn_sex_diff_abcd/dropbox/run_network_specific_svm.py -5 10 $1 $SLURM_ARRAY_TASK_ID
