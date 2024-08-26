import pandas as pd
from neuromaps import transforms
import nibabel as nib
import os


if __name__ == "__main__":
    os.environ["PATH"] += os.pathsep + '/Applications/workbench/bin_macosx64'
    pnc_gams_fslr = transforms.fsaverage_to_fslr(("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/gams_abs_sum_lh.gii", "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/gams_abs_sum_rh.gii"), target_density='32k')
    # pnc_gams_fslr = transforms.fsaverage_to_fslr(("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/GamSexAbssum_lh.gii", "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/GamSexAbssum_rh.gii"), target_density='32k')
    pnc_gams_fslr_full_data = list(pnc_gams_fslr[0].agg_data()) + list(pnc_gams_fslr[1].agg_data()) 
    pnc_gams_fslr_full_gii = nib.gifti.gifti.GiftiImage()
    pnc_gams_fslr_full_gii.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data=pnc_gams_fslr_full_data))
    nib.loadsave.save(pnc_gams_fslr_full_gii, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/Gam_abs_sum_fslr_test.gii")

