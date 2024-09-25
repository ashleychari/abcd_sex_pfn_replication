library(ciftiTools)

ciftiTools.setOption('wb_path', '/Users/ashfrana/Desktop/workbench')

# gam z scores
cifti_data2 <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/uncorrected_abs_sum_matrix_discovery.csv")
#cifti_data2 <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/uncorrected_abs_sum_matrix_replication.csv")
# svm weights
#cifti_data2 <- npyLoad("/Users/ashfrana/Desktop/code/ABCD GAMs replication/svm_072324_run/abs_sum_weight_brain_mat_discovery_haufe_100_runs_072324.npy")
#cifti_data2 <- npyLoad("/Users/ashfrana/Desktop/code/ABCD GAMs replication/svm_072324_run/abs_sum_weight_brain_mat_replication_haufe_100_runs_072324.npy")

# Include cifti_data2$XO for gams results and then take it off for svm results
cifti_lh2 <- as.numeric(cifti_data2$X0)[1:29696]
cifti_rh2 <- as.numeric(cifti_data2$X0)[29697:59412]

medialwall.mask.leftcortex <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.leftcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask
medialwall.mask.rightcortex <-read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.rightcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask

original_map_cifti <- as_cifti(cortexL = cifti_lh2, cortexL_mwall = medialwall.mask.leftcortex$V1, cortexR =  cifti_rh2, cortexR_mwall = medialwall.mask.rightcortex$V1)

map_medial_wall <- move_from_mwall(original_map_cifti)


write.csv(as.matrix(map_medial_wall), "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin tests/data/ABCD_data/gams_abs_sum_discovery_uncorrected_medial_wall.csv")

#write_cifti(xifti = map_medial_wall, cifti_fname = "/Users/ashfrana/Desktop/code/ABCD GAMs replication/medial_wall_maps/gams_replication_medial_wall_map.dscalar.nii")


test_nib <- read_cifti("/Users/ashfrana/Desktop/code/ABCD GAMs replication/test_engima_cifti/nibabel_gams_rep_uncorrected.dscalar.nii")
