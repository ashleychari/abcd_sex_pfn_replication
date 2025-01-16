import os

if __name__ == "__main__":
    disc_mats_path = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/matrices/discovery"
    rep_mats_path = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/matrices/replication"
    disc_mats = os.listdir("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/matrices/discovery")
    rep_mats = os.listdir("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/abs_sum_maps/matrices/replication")
    for mat in disc_mats:
        os.system(f"python3 write_effect_map_to_cifti.py {disc_mats_path}/{mat} cifti_files/discovery/{mat[0:len(mat)-4]}.dscalar.nii")
        #if mat != "pds_male_female_age_redo_z_mat.csv":
        os.system(f"python3 write_effect_map_to_cifti.py {rep_mats_path}/{mat} cifti_files/replication/{mat[0:len(mat)-4]}.dscalar.nii")