import os

if __name__ == "__main__":
    commands =[
        # Part A - discovery
        'sbatch puberty_shell_scripts/gams_puberty_timing_oSex_no_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_oSex_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_timing_oSex_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_oSex_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_oSex_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_oSex_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_oSex_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_oSex_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_oSex_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_oSex_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_oSex_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_oSex_no_age"',
        # Part A - replication
        'sbatch puberty_shell_scripts/gams_puberty_timing_oSex_no_age.sh replication puberty_data/replication/replication_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_oSex_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_timing_oSex_age.sh replication puberty_data/replication/replication_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_oSex_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_oSex_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_oSex_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_oSex_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_oSex_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_oSex_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_oSex_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_oSex_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_oSex_no_age"'
    ]

    print(len(commands))
    for command in commands:
        os.system(command)