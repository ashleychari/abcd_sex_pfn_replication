import numpy as np
import pandas as pd

def haufe_transform(features_path, weights_path, data_for_ridge_path, save_filename):
    features = np.load(features_path)
    targets = pd.read_csv(data_for_ridge_path)['sex']
    targets = [-1 if sex == "M" else 1 for sex in targets]
    coefs_matrix = np.load(weights_path)
    coefs = coefs_matrix.flatten()
    
    cov_x = []
    cov_y= []
    #compute Haufe-inverted feature weights
    cov_x = np.cov(np.transpose(features))
    cov_y = np.cov(targets)
    haufe_weights = np.matmul(cov_x,coefs)*(1/cov_y)
    
    haufe_brain_sex_matrix = []
    for i in range(1, 18):
        print(f"network {i}, start index: {(i-1) * 59412}, end index: { i * 59412}")
        haufe_brain_sex_matrix.append(haufe_weights[(i-1) * 59412 : i * 59412])

    haufe_brain_sex_matrix = np.array(haufe_brain_sex_matrix)
    np.save(save_filename, haufe_brain_sex_matrix)
    print("Job done!")

if __name__ == "__main__":
    discovery_data_for_ridge_path = '/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_removed_siblings.csv'
    discovery_coefs_matrix_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/w_brain_sex_matrix_one_run.npy'
    discovery_features_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery.npy'

    haufe_transform(discovery_features_path, discovery_coefs_matrix_path, discovery_data_for_ridge_path, "discovery_haufe_transformed_one_run_weights.npy")

    replication_data_for_ridge_path = '/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_removed_siblings.csv'
    replication_coefs_matrix_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/w_brain_sex_matrix_one_run_replication.npy'
    replication_features_path = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication.npy'

    haufe_transform(replication_features_path, replication_coefs_matrix_path, replication_data_for_ridge_path, "replication_haufe_transformed_one_run_weights.npy")



