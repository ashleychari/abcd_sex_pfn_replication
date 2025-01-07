import os

if __name__ == "__main__":
    disc_matrices_folders = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/individual_network_maps/discovery"
    rep_matrices_folders = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/individual_network_maps/replication"
    for folder in os.listdir(disc_matrices_folders):
        for i in range(1, 18):
            disc_filepath = f"{disc_matrices_folders}/{folder}/network_{i}.csv"
            rep_filepath = f"{rep_matrices_folders}/{folder}/network_{i}.csv"
            os.system(f"python3 /Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/write_effect_map_to_cifti.py {disc_filepath} {disc_matrices_folders}/{folder}/network_{i}.dscalar.nii")
            os.system(f"python3 /Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/write_effect_map_to_cifti.py {rep_filepath} {rep_matrices_folders}/{folder}/network_{i}.dscalar.nii")
    