from neuromaps.stats import compare_images
from neuromaps import nulls, datasets
import pandas as pd
import nibabel as nib
import numpy as np
import os
import sys
import random


# Note: under assumption that files being passed in are .gii files!
if __name__ == "__main__":
    print("Loading in files....")
    map_1_filename = sys.argv[1]
    map_2_filename = sys.argv[2]
    map_atlas = sys.argv[3]
    test_name = sys.argv[4]
    save_folder = sys.argv[5]

    # set density based on map
    if map_atlas == "fsaverage":
        density = '10k'
    else:
        density = '32k'

    # Spin null to compare between map_1 and map_2
    print("Making spin null....")
    rand_seed = random.randint(0, 200)
    map_1_null = nulls.alexander_bloch(map_1_filename, atlas=map_atlas, density=density, n_perm=1000, seed=rand_seed)
    np.save(f"{save_folder}/{test_name}_map1_null.npy", map_1_null)
    
    #comparing between map_1 and map_2
    print("Comparing maps....")
    corr_1_2, p_1_2 = compare_images(map_1_filename, map_2_filename, nulls=map_1_null)
    spin_test_df = pd.DataFrame()
    spin_test_df['test name'] = [test_name]
    spin_test_df['correlation'] = [corr_1_2]
    spin_test_df['pval'] = [p_1_2]
    spin_test_df.to_csv(f"{save_folder}/{test_name}_spin_test_results.csv")

    print(f"Spin test completed and results saved to {save_folder}/{test_name}_spin_test_results.csv")
    
    # Save command run into commands_run file
    command_run = f"python3 spin_test.py '{map_1_filename}' '{map_2_filename}' '{test_name}' '{save_folder}' \n" 
    f = open("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/commands_run.txt", "a+")
    f.write(command_run)
    f.close()



