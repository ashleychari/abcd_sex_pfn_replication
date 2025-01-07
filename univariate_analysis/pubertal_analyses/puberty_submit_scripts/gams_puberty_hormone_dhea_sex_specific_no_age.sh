#!/bin/bash
#SBATCH --propagate=NONE
#SBATCH --mem=50G
#SBATCH --time=1-00:00:00

set=$1
behavior_csv=$2
test_name=$3


singularity run --cleanenv \
    /cbica/projects/ash_pfn_sex_diff_abcd/software/containers/sex_differences_replication_0.0.3.sif \
    Rscript --save /cbica/projects/ash_pfn_sex_diff_abcd/dropbox/puberty_scripts_2/abcd_puberty_hormones.R $set $behavior_csv $test_name FALSE dhea FALSE