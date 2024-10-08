import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    # Create matrix for discovery set
    discovery_nonzero_index_df = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery_siblings_removed_nonzero_indices.csv')
    discovery_nonzero_indices = discovery_nonzero_index_df['nonzero_indices'].values
    results_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_roc_072324"
    brain_all_models1 = []
    brain_all_models2 = []
    for i in range(100):
        time = i+1
        discovery_filepath_coefs = f"{results_folder}/time_{time}/2foldcv_discovery_coefs.csv"
        discovery_coefs = pd.read_csv(discovery_filepath_coefs)
        brain_all_models1.append(discovery_coefs['fold_1_coefs'].values)
        brain_all_models2.append(discovery_coefs['fold_2_coefs'].values)

    brain_all_models1 = np.array(brain_all_models1)
    brain_all_models2 = np.array(brain_all_models2)
    brain_all_models = np.vstack((brain_all_models1, brain_all_models2))
    w_brain_sex_discovery = np.mean(brain_all_models, axis=0)

    # Create the matrix for the sum weights barplot
    w_brain_sex_all_discovery = np.zeros(17*59412)
    w_brain_sex_all_discovery[discovery_nonzero_indices] = w_brain_sex_discovery

    w_brain_sex_matrix_discovery = []
    for i in range(1, 18):
        w_brain_sex_matrix_discovery.append(w_brain_sex_all_discovery[(i - 1) * 59412: i * 59412])

    w_brain_sex_matrix_discovery = np.array(w_brain_sex_matrix_discovery)
    np.save(f"{results_folder}/w_brain_sex_matrix_100_times_discovery_072324.npy", w_brain_sex_matrix_discovery)


    # Create matrix for replication set
    replication_nonzero_index_df = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_siblings_removed_nonzero_indices.csv')
    replication_nonzero_indices = replication_nonzero_index_df['nonzero_indices'].values
    results_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_roc_072324"
    brain_all_models1 = []
    brain_all_models2 = []
    for i in range(100):
        time = i+1
        replication_filepath_coefs = f"{results_folder}/time_{time}/2foldcv_replication_coefs.csv"
        replication_coefs = pd.read_csv(replication_filepath_coefs)
        brain_all_models1.append(discovery_coefs['fold_1_coefs'].values)
        brain_all_models2.append(discovery_coefs['fold_2_coefs'].values)

    brain_all_models1 = np.array(brain_all_models1)
    brain_all_models2 = np.array(brain_all_models2)
    brain_all_models = np.vstack((brain_all_models1, brain_all_models2))
    w_brain_sex_replication = np.mean(brain_all_models, axis=0)

    # Create the matrix for the sum weights barplot
    w_brain_sex_all_replication = np.zeros(17*59412)
    w_brain_sex_all_replication[replication_nonzero_indices] = w_brain_sex_replication

    w_brain_sex_matrix_replication = []
    for i in range(1, 18):
        w_brain_sex_matrix_replication.append(w_brain_sex_all_replication[(i - 1) * 59412: i * 59412])

    w_brain_sex_matrix_replication = np.array(w_brain_sex_matrix_replication)
    np.save(f"{results_folder}/w_brain_sex_matrix_100_times_replication_072324.npy", w_brain_sex_matrix_replication)



    