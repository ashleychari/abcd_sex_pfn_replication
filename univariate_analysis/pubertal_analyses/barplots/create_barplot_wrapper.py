import os

if __name__ == "__main__":
    tests = ['hormone_ert_mf_oSex_age_mat.csv', 'hormone_ert_mf_oSex_no_age_mat.csv', 'pds_male_female_oSex_age_mat.csv',
             'pds_male_female_oSex_no_age_mat.csv', 'hormone_dhea_mf_oSex_age_mat.csv', 'hormone_dhea_mf_oSex_no_age_mat.csv']
    
    for test in tests:
        # Discovery matrices
        os.system(f"Rscript /Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/barplots/create_barplots.R discovery {test}")
        # Replication matrices
        os.system(f"Rscript /Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/barplots/create_barplots.R replication {test}")