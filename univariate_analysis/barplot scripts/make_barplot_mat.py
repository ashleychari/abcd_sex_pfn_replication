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
    create_abs_sum_mat("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/discovery/gams_sex_effects_siblings_removed", "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/discovery_barplot_all_networks_mat.csv", 
                       "discovery")
    create_abs_sum_mat("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/replication/gams_sex_effects_siblings_removed", "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/replication_barplot_all_networks_mat.csv", 
                       "replication")
