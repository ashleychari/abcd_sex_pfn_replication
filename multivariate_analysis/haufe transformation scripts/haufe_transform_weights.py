import numpy as np
import pandas as pd

def haufe_transform(features_path, weights_path, data_for_ridge_path, nonzero_indices, save_filename):
    features = np.load(features_path)
    targets = pd.read_csv(data_for_ridge_path)['sex']
    # Change this so Male=-1 and female=1
    targets = np.array([-1 if i == "M" else 1 for i in targets]) # set Male=1, female=-1
    coefs_matrix = np.load(weights_path)
    coefs = coefs_matrix.flatten()
    coefs = coefs[nonzero_indices]
    print(coefs.shape)
    
    cov_x = []
    cov_y= []
    #compute Haufe-inverted feature weights
    cov_x = np.cov(np.transpose(features))
    cov_y = np.cov(targets)
    haufe_weights = np.matmul(cov_x,coefs)*(1/cov_y)

    haufe_all_weights = np.zeros(17*59412)
    haufe_all_weights[nonzero_indices] = haufe_weights
    
    haufe_brain_sex_matrix = []
    for i in range(1, 18):
        print(f"network {i}, start index: {(i-1) * 59412}, end index: { i * 59412}")
        haufe_brain_sex_matrix.append(haufe_all_weights[(i-1) * 59412 : i * 59412])

    haufe_brain_sex_matrix = np.array(haufe_brain_sex_matrix)
    np.save(save_filename, haufe_brain_sex_matrix)
    print("Job done!")

if __name__ == "__main__":
    discovery_data_for_ridge_path = '/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_removed_siblings.csv'
    discovery_coefs_matrix_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final_2/w_brain_sex_matrix_100_times_discovery_final.npy'
    discovery_features_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery.npy'
    discovery_nonzero_index_df = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery_nonzero_indices.csv')
    discovery_nonzero_indices = discovery_nonzero_index_df['nonzero_indices'].values
    discovery_save_filename = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final_2/discovery_haufe_transformed_100_runs_weights_final.npy"
    haufe_transform(discovery_features_path, discovery_coefs_matrix_path, discovery_data_for_ridge_path, discovery_nonzero_indices, discovery_save_filename)

    replication_data_for_ridge_path = '/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_removed_siblings.csv'
    replication_coefs_matrix_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final_2/w_brain_sex_matrix_100_times_replication_final.npy'
    replication_features_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication.npy'
    replication_nonzero_index_df = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_nonzero_indices.csv')
    replication_nonzero_indices = replication_nonzero_index_df['nonzero_indices'].values
    replication_save_filename = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final_2/replication_haufe_transformed_100_runs_weights_final.npy"
    haufe_transform(replication_features_path, replication_coefs_matrix_path, replication_data_for_ridge_path, replication_nonzero_indices, replication_save_filename)



