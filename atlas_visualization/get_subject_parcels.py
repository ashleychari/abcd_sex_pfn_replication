import pandas as pd
import h5py
import numpy as np
import nibabel as nib

# Function to get a list of subjects for analysis
def get_subjects(discovery_data, replication_data):
    # Combine discovery and replication datasets into one DataFrame
    combined_data = pd.concat([discovery_data, replication_data], axis=0)

    # Extract subject keys based on sex
    male_subjects = combined_data[combined_data['sex'] == 'M']['subjectkey'].values
    female_subjects = combined_data[combined_data['sex'] == 'F']['subjectkey'].values

    # randomly select 4 subjects
    np.random.seed(90) # Set seed for reproducibility
    random_male_subjects = np.random.choice(male_subjects, 2)
    random_female_subject = np.random.choice(female_subjects, 2)

    # Combine selected subjects into a single list
    all_subjects = list(random_male_subjects) + list(random_female_subject)

    return all_subjects

# Function to save data to a CIFTI file format
def save_to_cifti(parcels, save_filename):
    # Load a template CIFTI file
    working_cifti_file = nib.load("/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/uncorrected_abs_sum_matrix_discovery.dscalar.nii")
    cifti_header = working_cifti_file.header

    # Create a new CIFTI image with the provided parcels and header
    new_cifti_img = nib.Cifti2Image(np.asanyarray(parcels), header=cifti_header)

    # Save the new CIFTI image to the specified filename
    nib.loadsave.save(new_cifti_img, save_filename)
    print("saved")

# Function to process and save soft parcel data for a given network and subjects
def get_soft_parcel(network, subjects):
    soft_parcel_networks = {}
    for i in range(len(subjects)):
        subject = subjects[i]
        # Construct the filename for the subject's data
        filename = f"/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/subject_data/sub-NDAR{subjects[i]}/IndividualParcel_Final_sbj1_comp17_alphaS21_1_alphaL300_vxInfo1_ard0_eta0/final_UV.mat"
        # Load subject's data from the .mat file
        subject_loadings = h5py.File(filename, "r")['#refs#']['c'][()] # (17, 59412) matrix
        # Extract network loadings for the specified network
        network_loadings = subject_loadings[network-1, :]
        # Store network loadings in a dictionary
        soft_parcel_networks[subject] = network_loadings

    # Save each subject's soft parcel data as a CIFTI file
    for subject, soft_parcel_network in soft_parcel_networks.items():
        soft_parcel_array = np.zeros((1, 59412))
        soft_parcel_array[0, :] = soft_parcel_network
        cifti_filename = f"/cbica/projects/ash_pfn_sex_diff_abcd/results/atlas_visualization/subject_networks_mf/network_{network}_softparcel_subjectid_{subject}.dscalar.nii"
        save_to_cifti(soft_parcel_array, cifti_filename)

    print(f"Files for network {network} saved to /cbica/projects/ash_pfn_sex_diff_abcd/results/atlas_visualization/subject_networks_mf/")

# Function to create hard parcel data from subject matrix
def create_hard_parcel_mat(subject_mat):
    hard_parcels = []
    # Determine the network with the highest loading for each vertex
    for i in range(59412):
        vertex_loadings = subject_mat[:, i]
        network_num = np.argmax(vertex_loadings) + 1
        hard_parcels.append(network_num)

    return hard_parcels

# Function to create a network visualization array
def create_network_viz(network, hard_parcels):
    encoded_vec = []
     # Initialize an array to hold the network visualization
    encoded_array = np.zeros((1, 59412))

    # One hot encode network from parcels
    for parcel in hard_parcels:
        if parcel == network:
            encoded_vec.append(1)
        else:
            encoded_vec.append(0)
    
    encoded_array[0, :] = encoded_vec
    return encoded_array

# Function to process and save hard parcel data for a given network and subjects
def get_hard_parcel(network, subjects):
    subject_hardparcel_dict = {}
    for i in range(len(subjects)):
        subject = subjects[i]
        # Construct the filename for the subject's data
        filename = f"/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/subject_data/sub-NDAR{subjects[i]}/IndividualParcel_Final_sbj1_comp17_alphaS21_1_alphaL300_vxInfo1_ard0_eta0/final_UV.mat"
        # Load subject's data from the .mat file
        subject_loadings = h5py.File(filename, "r")['#refs#']['c'][()] # (17, 59412) matrix
        # Create hard parcels from subject data
        hard_parcels = create_hard_parcel_mat(subject_loadings)
        # Create network visualization array
        network_array = create_network_viz(network, hard_parcels)
        subject_hardparcel_dict[subject] = network_array

    # Save each subject's hard parcel data as a CIFTI file
    for subject, hard_parcel_network in subject_hardparcel_dict.items():
        hard_parcel_filename = f'/cbica/projects/ash_pfn_sex_diff_abcd/results/atlas_visualization/subject_networks_mf/network_{network}_hardparcel_subjectid_{subject}.dscalar.nii'
        save_to_cifti(hard_parcel_network, hard_parcel_filename)

    print(f"Files for network {network} saved to /cbica/projects/ash_pfn_sex_diff_abcd/results/atlas_visualization/subject_networks_mf/")


if __name__ == "__main__":
    # Load discovery and replication datasets
    discovery_dataset = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_siblings_removed_071524.csv')
    replication_dataset = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_siblings_removed_071524.csv')

    # Get a random sample of 4 subjects - equal amounts of M/F subjects
    random_4_subjects = get_subjects(discovery_dataset, replication_dataset)

    # Network 3 - soft and hard parcel
    get_soft_parcel(3, random_4_subjects)
    get_hard_parcel(3, random_4_subjects)

    # Network 4 - soft and hard parcel
    get_soft_parcel(4, random_4_subjects)
    get_hard_parcel(4, random_4_subjects)

    # Network 12 - soft and hard parcel
    get_soft_parcel(12, random_4_subjects)
    get_hard_parcel(12, random_4_subjects)



