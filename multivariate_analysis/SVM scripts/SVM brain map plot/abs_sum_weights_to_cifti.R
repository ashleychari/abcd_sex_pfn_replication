library(ciftiTools)
library(RcppCNPy)

#cifti_data2 <- npyLoad("/Users/ashfrana/Desktop/code/ABCD GAMs replication/abs_sum_weight_brain_mat.npy")
#cifit_data2 <- npyLoad("/Users/ashfrana/Desktop/code/ABCD GAMs replication/abs_sum_weight_brain_mat_replication.npy")
#cifti_data2 <- npyLoad("/Users/ashfrana/Desktop/code/ABCD GAMs replication/svm_correct_results/abs_sum_weight_brain_mat_discovery_haufe_100_runs.npy")
cifti_data2 <- npyLoad("/Users/ashfrana/Desktop/code/ABCD GAMs replication/svm_correct_results/abs_sum_weight_brain_mat_replication_haufe_100_runs.npy")
cifti_lh2 <- as.numeric(cifti_data2)[0:29696]
cifti_rh2 <- as.numeric(cifti_data2)[29697:59412]

medialwall.mask.leftcortex <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.leftcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask
medialwall.mask.rightcortex <-read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.rightcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask

unicorn.dscalar <- as_cifti(cortexL = cifti_lh2, cortexL_mwall = medialwall.mask.leftcortex$V1, cortexR =  cifti_rh2, cortexR_mwall = medialwall.mask.rightcortex$V1)


# Write a CIFTI file --------------------------------------------
write_cifti(xifti = unicorn.dscalar, cifti_fname = "/Users/ashfrana/Desktop/code/ABCD GAMs replication/svm_correct_results/svm_abs_sum_weights_replication.dscalar.nii")
ciftiTools.setOption('wb_path', '/Users/ashfrana/Desktop/workbench')