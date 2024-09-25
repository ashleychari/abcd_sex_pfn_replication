import enigmatoolbox
import nibabel as nib
import pandas as pd
import numpy as np
import sys

# Take in data to convert and save filename
if __name__ == "__main__":
    data_filepath = sys.argv[1]
    save_filepath = sys.argv[2]
    is_PNC = sys.argv[3]

    if data_filepath.split(".")[1] == "csv" and is_PNC == "N":
        abs_sum_mat = pd.read_csv(data_filepath)
        print(abs_sum_mat)
    elif data_filepath.split(".")[1] == "csv" and is_PNC == "Y":
        abs_sum_mat = pd.read_csv(data_filepath, header=None)
    else:
        abs_sum_mat = np.load(data_filepath)

    data = nib.gifti.gifti.GiftiImage()
    if "Unnamed: 0" in abs_sum_mat.columns:
        abs_sum_mat = abs_sum_mat.drop(["Unnamed: 0"], axis=1)

    if data_filepath.split(".")[1] == "csv":
        abs_sum_float32 = [np.float32(val) for val in abs_sum_mat[abs_sum_mat.columns[0]].values]
    else:
        abs_sum_float32 = [np.float32(val) for val in abs_sum_mat]

    print(len(abs_sum_float32))

    data.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data=abs_sum_float32))

    nib.save(data, save_filepath)

    print(f"File saved to {save_filepath}!")
    
    # Save command run into commands_run file
    command_run = f"python3 convert_to_gifti.py '{data_filepath}' '{save_filepath}' {is_PNC}\n" 
    f = open("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/commands_run.txt", "a+")
    f.write(command_run)
    f.close()


