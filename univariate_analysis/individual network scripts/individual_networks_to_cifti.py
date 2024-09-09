import os

if __name__ == "__main__":
    for i in range(1, 18):
        os.system(f"python3 '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/workbench setup scripts/write_effect_map_to_cifti.py' '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/individual network scripts/individual_network_matrices/individual_network_{i}_mat.csv' '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/network_{i}_sig_discovery.dscalar.nii'")