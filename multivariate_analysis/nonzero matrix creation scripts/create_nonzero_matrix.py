import pandas as pd
import numpy as np
import h5py
import sys

def create_matrix(data_for_ridge, save_filename):
    # Get all the subject data
    subject_ids = data_for_ridge['subjectkey']
    features = []
    print("Adding features....")
    # Flatten loadings matrices for each subject and add to features
    print(f"num subject ids {len(subject_ids)}")
    for i in range(len(subject_ids)):
        filename = f"/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/subject_data/sub-NDAR{subject_ids[i]}/IndividualParcel_Final_sbj1_comp17_alphaS21_1_alphaL300_vxInfo1_ard0_eta0/final_UV.mat"
        subject_loadings = h5py.File(filename, "r")['#refs#']['c'][()] # (17, 59412) matrix
        flattened_array = subject_loadings.flatten() #1010004 1-dimensional array
        print(f"i: {i}, flattened_array.shape: {flattened_array.shape}")
        features.append(flattened_array)

    # Identify nonzero features and subset dataframe for them
    features = np.array(features)
    features_nonzero_array = features[:, features.any(0)]
    print(f"features_nonzero_array.shape: {features_nonzero_array.shape}")

    # Save nonzero indices
    save_nonzero_inds =  f"/cbica/projects/ash_pfn_sex_diff_abcd/results/{save_filename}_nonzero_indices.csv"
    nonzero_inds_df = pd.DataFrame()
    nonzero_inds_df['nonzero_indices'] = np.nonzero(np.any(features != 0, axis=0))[0]
    nonzero_inds_df.to_csv(save_nonzero_inds)
    print(f"Nonzero indices saved to {save_nonzero_inds}")

    
    save_array_filepath = f"/cbica/projects/ash_pfn_sex_diff_abcd/results/{save_filename}.npy"
    np.save(save_array_filepath, features_nonzero_array)
    print(f"Matrix of shape {features.shape} saved to {save_array_filepath}!")


if __name__ == "__main__":
    filename = sys.argv[1]
    save_filename = sys.argv[2]
    data_for_ridge = pd.read_csv(filename)
    create_matrix(data_for_ridge, save_filename)
    


    





    
        


