import pandas as pd
import numpy as np
import nibabel as nib

# Function to create a hard parcel matrix from the group average matrix
def create_hard_parcel_mat(group_avg_mat):
    hard_parcels = []
    for i in range(59412):
        # Extract the loadings for the current vertex across all networks
        vertex_loadings = group_avg_mat[:, i]
        # Determine the network with the maximum loading for this vertex
        network_num = np.argmax(vertex_loadings) + 1
        hard_parcels.append(network_num)

    # Convert the list of hard parcels to a numpy array with shape (1, 59412)
    hard_parcels_array = np.zeros((1, 59412))
    hard_parcels_array[0,:] = hard_parcels

    return hard_parcels_array, hard_parcels

def create_network_viz(network, hard_parcels):
    encoded_vec = []
     # Initialize array to hold the network visualization
    encoded_array = np.zeros((1, 59412))

    # Create a binary vector where each position is 1 if it matches the specified network, otherwise 0
    for parcel in hard_parcels:
        if parcel == network:
            encoded_vec.append(1)
        else:
            encoded_vec.append(0)
    
    encoded_array[0, :] = encoded_vec
    return encoded_array

# Function to save the data as a CIFTI file
def save_to_cifti(hard_parcels, save_filename):
    # Load a CIFTI template file to use its header
    working_cifti_file = nib.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/hardparcel_group.dlabel.nii")
    cifti_header = working_cifti_file.header
    # Create a new CIFTI image with the provided data and header
    new_cifti_img = nib.Cifti2Image(np.asanyarray(hard_parcels), header=cifti_header)
    # Save the new CIFTI image to the specified filename
    nib.loadsave.save(new_cifti_img, save_filename)
    print("saved")

if __name__ == "__main__":
    # Define the filename for saving the hard parcels matrix
    save_filename = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/hardparcels_group_080924.dlabel.nii"

    # Load the group average matrix from a .npy file
    group_avg_mat = np.load("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/group_average_matrix.npy")

    # Create the hard parcel matrix and extract the list of hard parcels
    hard_parcels_array, hard_parcels = create_hard_parcel_mat(group_avg_mat)

    # Save the hard parcels matrix to a CIFTI file
    save_to_cifti(hard_parcels_array, save_filename)

    # Generate and save hard parcel visualizations for networks 3, 4, and 12
    save_network_3_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/network_3_hardparcels.dlabel.nii'
    network_3_viz = create_network_viz(3, hard_parcels)
    save_to_cifti(network_3_viz, save_network_3_filename)

    save_network_4_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/network_4_hardparcels.dlabel.nii'
    network_4_viz = create_network_viz(4, hard_parcels)
    save_to_cifti(network_4_viz, save_network_4_filename)

    save_network_12_filename = '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/atlas_visualization/network_12_hardparcels.dlabel.nii'
    network_12_viz = create_network_viz(12, hard_parcels)
    save_to_cifti(network_12_viz, save_network_12_filename)


