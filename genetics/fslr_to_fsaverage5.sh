workbench="/Users/ashfrana/Desktop/workbench/bin_macosx64"

file_to_sep=$1
save_filename=$2


${workbench}/wb_command -cifti-separate ${file_to_sep} COLUMN -metric CORTEX_LEFT "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/"${save_filename}"_LH.fslr32k.func.gii" 
${workbench}/wb_command -cifti-separate ${file_to_sep}  COLUMN -metric CORTEX_RIGHT "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/"${save_filename}"_RH.fslr32k.func.gii"

#Resample S-A axis from fslr mesh to fsaverage5 mesh

${workbench}/wb_command -metric-resample "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/"${save_filename}"_LH.fslr32k.func.gii" "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fs_LR-deformed_to-fsaverage.L.sphere.32k_fs_LR.surf.gii" "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii" ADAP_BARY_AREA "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/"${save_filename}"_LH.fsaverage5.func.gii" -area-metrics "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fs_LR.L.midthickness_va_avg.32k_fs_LR.shape.gii" "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fsaverage5.L.midthickness_va_avg.10k_fsavg_L.shape.gii"

${workbench}/wb_command -metric-resample "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/SensorimotorAssociation_Axis_RH.fslr32k.func.gii" "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fs_LR-deformed_to-fsaverage.R.sphere.32k_fs_LR.surf.gii" "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii" ADAP_BARY_AREA "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/"${save_filename}"_RH.fsaverage5.func.gii" -area-metrics "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fs_LR.R.midthickness_va_avg.32k_fs_LR.shape.gii" "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/fsaverage5.R.midthickness_va_avg.10k_fsavg_R.shape.gii"

# Save the command run in commands_run.txt
command="./fslr_to_fsaverage5.sh $file_to_sep $save_filename"
echo $command >> "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/commands_run.txt"
