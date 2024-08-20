import nibabel as nib
import pandas as pd
import numpy as np

def soft_parcel_network(network, group_avg_mat):
    final_parcels = np.zeros((1, 59412))
    network_soft_parcel = group_avg_mat[network-1, :]
    final_parcels[0, :] = network_soft_parcel
    return final_parcels

def save_to_cifti(soft_parcels, save_filename):
    working_cifti_file = nib.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/uncorrected_abs_sum_matrix_discovery.dscalar.nii")
    cifti_header = working_cifti_file.header
    new_cifti_img = nib.Cifti2Image(np.asanyarray(soft_parcels), header=cifti_header)
    nib.loadsave.save(new_cifti_img, save_filename)
    print("saved")

if __name__ == "__main__":
    group_avg_mat = np.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_average_matrix.npy")

    save_network_3_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/soft_parcels/network_3_softparcels.dscalar.nii'
    network_3_viz = soft_parcel_network(3, group_avg_mat)
    save_to_cifti(network_3_viz, save_network_3_filename)

    save_network_4_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/soft_parcels/network_4_softparcels.dscalar.nii'
    network_4_viz = soft_parcel_network(4, group_avg_mat)
    save_to_cifti(network_4_viz, save_network_4_filename)

    save_network_12_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/soft_parcels/network_12_softparcels.dscalar.nii'
    network_12_viz = soft_parcel_network(12, group_avg_mat)
    save_to_cifti(network_12_viz, save_network_12_filename)

