import pandas as pd
import numpy as np
import nibabel as nib

def create_hard_parcel_mat(group_avg_mat):
    hard_parcels = []
    for i in range(59412):
        vertex_loadings = group_avg_mat[:, i]
        network_num = np.argmax(vertex_loadings) + 1
        hard_parcels.append(network_num)

    hard_parcels_array = np.zeros((1, 59412))
    hard_parcels_array[0,:] = hard_parcels

    return hard_parcels_array, hard_parcels

def create_network_viz(network, hard_parcels):
    encoded_vec = []
    encoded_array = np.zeros((1, 59412))
    for parcel in hard_parcels:
        if parcel == network:
            encoded_vec.append(1)
        else:
            encoded_vec.append(0)
    
    encoded_array[0, :] = encoded_vec
    return encoded_array

def save_to_cifti(hard_parcels, save_filename):
    working_cifti_file = nib.load("/Users/ashfrana/Desktop/code/ABCD GAMs replication/univariate_analysis/unthresholded_abs_sum_replication_071624.dscalar.nii")
    cifti_header = working_cifti_file.header
    new_cifti_img = nib.Cifti2Image(np.asanyarray(hard_parcels), header=cifti_header)
    nib.loadsave.save(new_cifti_img, save_filename)
    print("saved")

if __name__ == "__main__":
    save_filename = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/hardparcels_group_080924.dscalar.nii"
    group_avg_mat = np.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_average_matrix.npy")
    hard_parcels_array, hard_parcels = create_hard_parcel_mat(group_avg_mat)
    save_to_cifti(hard_parcels_array, save_filename)

    save_network_3_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/network_3_hardparcels.dscalar.nii'
    network_3_viz = create_network_viz(3, hard_parcels)
    save_to_cifti(network_3_viz, save_network_3_filename)

    save_network_4_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/network_4_hardparcels.dscalar.nii'
    network_4_viz = create_network_viz(4, hard_parcels)
    save_to_cifti(network_4_viz, save_network_4_filename)

    save_network_12_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/network_12_hardparcels.dscalar.nii'
    network_12_viz = create_network_viz(12, hard_parcels)
    save_to_cifti(network_12_viz, save_network_12_filename)


