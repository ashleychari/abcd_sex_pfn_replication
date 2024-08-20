import h5py
import pandas as pd
import numpy as np

# combine the discovery and replication sets into one
def combine_data(discovery_set, replication_set):
    combined_df = pd.concat([discovery_set, replication_set], axis=0)
    return combined_df

# Make network specific matrices made up of all the subjects
def make_network_specific_mats(subject_data):
    network_matrices_dict = {}
    subjects = subject_data['subjectkey'].values

    # Read in 
    for i in range(17):
        print(f"Making network {i+1} matrix....")
        network_array = np.zeros((len(subjects), 59412))
        for j in range(len(subjects)):
            filename = f"/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/subject_data/sub-NDAR{subjects[j]}/IndividualParcel_Final_sbj1_comp17_alphaS21_1_alphaL300_vxInfo1_ard0_eta0/final_UV.mat"
            subject_loadings = h5py.File(filename, "r")['#refs#']['c'][()] # (17, 59412) matrix
            subject_loadings_network = subject_loadings[i, :]
            network_array[j,] = subject_loadings_network

        network_matrices_dict[i] = network_array

    return network_matrices_dict

# Get the group averages for each of these matrices and then make one big matrix
def get_group_averages(network_matrices_dict, save_filename):
    group_avg_matrix = np.zeros((17, 59412))
    print(f"Averaging network matrices into one matrix....")
    for network, network_matrix in network_matrices_dict.items():
        # get the average for each vertex
        avg_network_mat = network_matrix.mean(axis=0)
        group_avg_matrix[network] = avg_network_mat

    np.save(save_filename, group_avg_matrix)


if __name__ == "__main__":
    discovery_dataset = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_siblings_removed_071524.csv')
    replication_dataset = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_siblings_removed_071524.csv')
    save_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/atlas_visualization/group_average_matrix.npy'
    full_sample = combine_data(discovery_dataset, replication_dataset)
    network_matrix_dict = make_network_specific_mats(full_sample)
    get_group_averages(network_matrix_dict, save_filename)
