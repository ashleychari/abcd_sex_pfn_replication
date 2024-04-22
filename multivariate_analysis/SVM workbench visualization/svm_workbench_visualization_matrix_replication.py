import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    replication_nonzero_index_df = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_nonzero_indices.csv')
    replication_nonzero_indices = replication_nonzero_index_df['nonzero_indices'].values
    results_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/"
    brain_all_models1 = []
    brain_all_models2 = []
    
    replication_filepath_coefs = f"{results_folder}/2foldcv_replication_coefs.csv"
    replication_coefs = pd.read_csv(replication_filepath_coefs)
    brain_all_models1.append(replication_coefs['fold_1_coefs'].values)
    brain_all_models2.append(replication_coefs['fold_2_coefs'].values)

    brain_all_models1 = np.array(brain_all_models1)
    brain_all_models2 = np.array(brain_all_models2)
    brain_all_models = np.vstack((brain_all_models1, brain_all_models2))
    w_brain_sex = np.mean(brain_all_models, axis=0)
    print(w_brain_sex.shape)
    print(replication_nonzero_indices.shape)
    # Create the matrix for the sum weights barplot
    w_brain_sex_all = np.zeros(17*59412)
    w_brain_sex_all[replication_nonzero_indices] = w_brain_sex

    w_brain_sex_matrix = []
    for i in range(1, 18):
        print(f"network {i}, start index: {(i-1) * 59412}, end index: { i * 59412}")
        w_brain_sex_matrix.append(w_brain_sex_all[(i-1) * 59412 : i * 59412])

    w_brain_sex_matrix = np.array(w_brain_sex_matrix)
    np.save(f"{results_folder}/w_brain_sex_matrix_one_run_replication.npy", w_brain_sex_matrix)
    print("Job done!")