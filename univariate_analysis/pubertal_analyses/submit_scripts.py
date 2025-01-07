import os

if __name__ == "__main__":
    commands =[
        # Part A - discovery
        'sbatch puberty_shell_scripts/gams_puberty_timing_no_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_timing_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_no_age"',
        # Part A - replication
        'sbatch puberty_shell_scripts/gams_puberty_timing_no_age.sh replication puberty_data/replication/replication_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_timing_age.sh replication puberty_data/replication/replication_behavior_pds_p_category_subjectkey_M_F.csv "pds_male_female_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_mf_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_M_F.csv "hormone_ert_mf_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_mf_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_M_F.csv "hormone_dhea_mf_no_age"',
        # Part B  - pds sex specific discovery
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_ss_female_category_subjectkey_F.csv "pds_female_only_age" "F"',
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_no_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_ss_female_category_subjectkey_F.csv "pds_female_only_no_age" "F"',
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_ss_male_category_subjectkey_M.csv "pds_male_only_age" "M"',
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_no_age.sh discovery puberty_data/discovery/discovery_behavior_pds_p_ss_male_category_subjectkey_M.csv "pds_male_only_no_age" "M"',
        # Part B - pds sex specific replication
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_age.sh replication puberty_data/replication/replication_behavior_pds_p_ss_female_category_subjectkey_F.csv "pds_female_only_age" "F"',
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_no_age.sh replication puberty_data/replication/replication_behavior_pds_p_ss_female_category_subjectkey_F.csv "pds_female_only_no_age" "F"',
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_age.sh replication puberty_data/replication/replication_behavior_pds_p_ss_male_category_subjectkey_M.csv "pds_male_only_age" "M"',
        'sbatch puberty_shell_scripts/gams_puberty_pds_sex_specific_no_age.sh replication puberty_data/replication/replication_behavior_pds_p_ss_male_category_subjectkey_M.csv "pds_male_only_no_age" "M"',
        # Part B - hse hormone female only discovery
        'sbatch puberty_shell_scripts/gams_puberty_hormone_hse_female_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_hse_mean_subjectkey_F.csv "hormone_hse_female_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_hse_female_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_hse_mean_subjectkey_F.csv "hormone_hse_female_only_no_age"',
        # Part B - hse horomone female only replication
        'sbatch puberty_shell_scripts/gams_puberty_hormone_hse_female_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_hse_mean_subjectkey_F.csv "hormone_hse_female_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_hse_female_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_hse_mean_subjectkey_F.csv "hormone_hse_female_only_no_age"',
        # Part B - ert hormone sex specific discovery
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_F.csv "hormone_ert_female_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_F.csv "hormone_ert_female_only_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_M.csv "hormone_ert_male_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_ert_mean_subjectkey_M.csv "hormone_ert_male_only_no_age"',
        # Part B - ert horomone sex specific replication
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_F.csv "hormone_ert_female_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_F.csv "hormone_ert_female_only_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_M.csv "hormone_ert_male_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_ert_sex_specific_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_ert_mean_subjectkey_M.csv "hormone_ert_male_only_no_age"',
        # Part C - dhea hormone sex specific discovery
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_F.csv "hormone_dhea_female_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_F.csv "hormone_dhea_female_only_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_M.csv "hormone_dhea_male_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_no_age.sh discovery puberty_data/discovery/discovery_behavior_hormone_scr_dhea_mean_subjectkey_M.csv "hormone_dhea_male_only_no_age"',
        # Part C - dhea horomone sex specific replication
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_F.csv "hormone_dhea_female_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_F.csv "hormone_dhea_female_only_no_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_M.csv "hormone_dhea_male_only_age"',
        'sbatch puberty_shell_scripts/gams_puberty_hormone_dhea_sex_specific_no_age.sh replication puberty_data/replication/replication_behavior_hormone_scr_dhea_mean_subjectkey_M.csv "hormone_dhea_male_only_no_age"'
    ]

    print(len(commands))
    for command in commands:
        os.system(command)