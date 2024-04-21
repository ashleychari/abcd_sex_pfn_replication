#!/bin/bash

singularity run --cleanenv \
    /cbica/projects/ash_pfn_sex_diff_abcd/software/containers/sex_differences_replication_0.0.3.sif \
    Rscript --save /cbica/projects/ash_pfn_sex_diff_abcd/dropbox/abcd_univariate_analysis.R