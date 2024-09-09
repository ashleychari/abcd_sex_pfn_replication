import nibabel as nib
import pandas as pd
import numpy as np

# Function to create a soft parcel network visualization
def soft_parcel_network(network, group_avg_mat):
    # Initialize an array to store the final parcels data
    final_parcels = np.zeros((1, 59412))
    # Extract the soft parcel data for the specified network from the group average matrix
    network_soft_parcel = group_avg_mat[network-1, :]
    # Assign the extracted data to the final parcels array
    final_parcels[0, :] = network_soft_parcel
    return final_parcels

# Function to save the soft parcel network visualization to a CIFTI file
def save_to_cifti(soft_parcels, save_filename):
    # Load a CIFTI template file to use its header
    working_cifti_file = nib.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/uncorrected_abs_sum_matrix_discovery.dscalar.nii")
    cifti_header = working_cifti_file.header
    # Create a new CIFTI image with the provided soft parcels data and header
    new_cifti_img = nib.Cifti2Image(np.asanyarray(soft_parcels), header=cifti_header)
    # Save the new CIFTI image to the specified filename
    nib.loadsave.save(new_cifti_img, save_filename)
    print("saved")

if __name__ == "__main__":
    # Load the group average matrix from a .npy file
    group_avg_mat = np.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_average_matrix.npy")

    # Generate and save soft parcel visualizations for networks 3, 4, and 12
    save_network_3_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_level_parcels_updated/soft_parcels/network_3_softparcels.dscalar.nii'
    network_3_viz = soft_parcel_network(3, group_avg_mat)
    save_to_cifti(network_3_viz, save_network_3_filename)

    save_network_4_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_level_parcels_updated/soft_parcels/network_4_softparcels.dscalar.nii'
    network_4_viz = soft_parcel_network(4, group_avg_mat)
    save_to_cifti(network_4_viz, save_network_4_filename)

    save_network_12_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_level_parcels_updated/soft_parcels/network_12_softparcels.dscalar.nii'
    network_12_viz = soft_parcel_network(12, group_avg_mat)
    save_to_cifti(network_12_viz, save_network_12_filename)

