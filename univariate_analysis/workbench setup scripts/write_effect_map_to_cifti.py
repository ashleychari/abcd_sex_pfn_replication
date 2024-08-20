import nibabel as nib
import pandas as pd
import sys
import os
import numpy as np

if __name__ == "__main__":
    # Get the filename of the data to be converted and the filename to be saved to
    input_filename = sys.argv[1]
    save_filename = sys.argv[2]

    # Read in input_filename
    data_to_be_converted = pd.read_csv(input_filename)
    print(f"Value from file to be converted: {data_to_be_converted['0'].values[0]}")

    # Use working cifti header to create image with new data
    working_cifti_file = nib.load("/Users/ashfrana/Desktop/code/ABCD GAMs replication/univariate_analysis/unthresholded_abs_sum_replication_071624.dscalar.nii")
    cifti_header = working_cifti_file.header

    # Create and save new cifti image with the inputted data
    new_cifti_img = nib.Cifti2Image(np.asanyarray(data_to_be_converted.T), header=cifti_header)
    print(f"Value in new cifti image before saving: {new_cifti_img.get_fdata()[0][0]}")
    nib.loadsave.save(new_cifti_img, save_filename)

    #print(f"Converted file saved to {save_filename}!")

    # Save command run into commands_run file
    command_run = f"python3 write_effect_map_to_cifti.py '{input_filename}' '{save_filename}'\n" 
    f = open("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/commands_run.txt", "a+")
    f.write(command_run)
    f.close()


    reloaded_img = nib.load(save_filename)
    print(f"Value from converted image reloaded: {reloaded_img.get_fdata()[0][0]}")

