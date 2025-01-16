import pandas as pd
import os

def create_abs_sum_mat(results_folder, save_filename, sample_type):
    all_z_vectors = pd.DataFrame()
    network = 1
    for i in range(17):
        if sample_type == "discovery":
            result_filename = f"{results_folder}/SexEffect_AtlasLoading_Discovery_17_Network_{network}.csv"
        else:
            result_filename = f"{results_folder}/SexEffect_AtlasLoading_Replication_17_Network_{network}.csv"
        network_matrix = pd.read_csv(result_filename)
        network_matrix = network_matrix['Gam_Z_FDR_Sig_Vector_All']
        all_z_vectors[network] = network_matrix
        network += 1
        
    
    all_z_vectors.T.to_csv(save_filename, index=False)
    print("Job Done!")


if __name__ == "__main__":
    discovery_tests = os.listdir("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/discovery")
    replication_tests = os.listdir("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/replication")
    disc_results_folder = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/discovery"
    disc_save_folder = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/barplots/matrices_redo/discovery"
    for disc_test in discovery_tests:
        results_folder = f"{disc_results_folder}/{disc_test}"
        filename = f"{disc_save_folder}/{disc_test}_mat.csv"
        create_abs_sum_mat(results_folder, filename, "discovery")

    rep_results_folder = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/puberty_analyses_2/replication"
    rep_save_folder = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/barplots/matrices_redo/replication"
    for rep_test in replication_tests:
        results_folder = f"{rep_results_folder}/{rep_test}"
        filename = f"{rep_save_folder}/{rep_test}_mat.csv"
        create_abs_sum_mat(results_folder, filename, "replication")
    
