import os

if __name__ == "__main__":
    results_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/network_specific_models_112124"
    for i in range(1, 18):
        if not os.path.exists(f"{results_folder}/network_{i}"):
            os.mkdir(f"{results_folder}/network_{i}")
        
        os.system(f"sbatch network_specific_models_slurm.sh {i}")

