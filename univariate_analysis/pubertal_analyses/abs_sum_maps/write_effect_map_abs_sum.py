import pandas as pd
import os

def create_abs_sum_mat(results_folder, sample_type, save_filename):
    all_z_vectors = pd.DataFrame()
    network = 1
    for i in range(17):
        if sample_type == "discovery":
            result_filename = f"{results_folder}/SexEffect_AtlasLoading_Discovery_17_Network_{network}.csv"
        else:
            result_filename = f"{results_folder}/SexEffect_AtlasLoading_Replication_17_Network_{network}.csv"
        network_matrix = pd.read_csv(result_filename)
        network_matrix = network_matrix['Gam_Z_Vector_All']
        all_z_vectors[f"network_{network}"] = abs(network_matrix)
        network += 1
        
    
    network_abs_sum = all_z_vectors.sum(axis=1)
    network_abs_sum.to_csv(save_filename, index=False)
    print("Job Done!")


if __name__ == "__main__":
    tests = os.listdir("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/discovery")
    #tests = ['hormone_ert_mf_age', 'pds_male_female_no_age', 'pds_male_female_age']
    for test in tests:
        disc_results_folder = f"/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/discovery/{test}"
        rep_results_folder = f"/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/replication/{test}"
        disc_save_filename = f"/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/matrices/discovery/{test}_z_mat.csv"
        rep_save_filename = f"/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/matrices/replication/{test}_z_mat.csv"
        create_abs_sum_mat(disc_results_folder, "discovery", disc_save_filename)
        create_abs_sum_mat(rep_results_folder, "replication", rep_save_filename)



