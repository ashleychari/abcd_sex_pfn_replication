import h5py
import numpy as np
import os
import pandas as pd


if __name__ == "__main__":
    # loading folder and folder where you store converted mat files
    ATLAS_LOADING_FOLDER = "/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/subject_data"
    STORAGE_FOLDER = "/cbica/projects/ash_pfn_sex_diff_abcd/results/converted_matrices"

    # read in behavior dataframe to get the subject keys (need to change this reference)
    subject_folders = os.listdir(ATLAS_LOADING_FOLDER)
    # behavior_df = pd.read_csv("/cbica/projects/abcd_pfn_sex_diff/dropbox/data_for_ridge_032822.csv")
    # subject_keys = behavior_df['subjectkey'].values

    # Check if storage folder exists
    if not os.path.isdir(STORAGE_FOLDER):
        os.mkdir(STORAGE_FOLDER)

    # Go through all of the subject keys and get their corresponding loading file
    for i in range(len(subject_folders)):
        subject_key = subject_folders[i].strip("/")[8:19]
        atlas_loading_file = f"{ATLAS_LOADING_FOLDER}/sub-NDAR{subject_key}/IndividualParcel_Final_sbj1_comp17_alphaS21_1_alphaL300_vxInfo1_ard0_eta0/final_UV.mat"
        hfd5_file = h5py.File(atlas_loading_file)
        
        # create numpy file and save data into numpy file
        df = hfd5_file['#refs#']['c'][()].T

        # Check if subject folder is created
        if not os.path.isdir(f"{STORAGE_FOLDER}/sub-NDAR{subject_key}/"):
            os.mkdir(f"{STORAGE_FOLDER}/sub-NDAR{subject_key}/")

        # Check if parcel folder is created
        if not os.path.isdir(f"{STORAGE_FOLDER}/sub-NDAR{subject_key}/parcel_1by1/"):
            os.mkdir(f"{STORAGE_FOLDER}/sub-NDAR{subject_key}/parcel_1by1/")

        # Save array to pandas df/csv file
        pandas_save_filename = f"{STORAGE_FOLDER}/sub-NDAR{subject_key}/parcel_1by1/converted_UV_mat.csv"
        df = pd.DataFrame(df)
        df.to_csv(pandas_save_filename, index=False)
        print(f"{pandas_save_filename} saved!")

    print("Job done!")


    