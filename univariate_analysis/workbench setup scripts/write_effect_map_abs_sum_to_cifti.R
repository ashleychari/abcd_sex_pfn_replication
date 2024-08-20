library(ciftiTools)

cifti_data2 <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/unthresholded_abs_sum_matrix_replication_2.csv")

cifti_lh2 <- as.numeric(cifti_data2$X0)[0:29696]
cifti_rh2 <- as.numeric(cifti_data2$X0)[29697:59412]

medialwall.mask.leftcortex <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.leftcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask
medialwall.mask.rightcortex <-read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.rightcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask

unicorn.dscalar <- as_cifti(cortexL = cifti_lh2, cortexL_mwall = medialwall.mask.leftcortex$V1, cortexR =  cifti_rh2, cortexR_mwall = medialwall.mask.rightcortex$V1)

# NOTE: We ended up not using this file for cifti creation because there is something wrong with the write_cifti function!!!!!
# Write a CIFTI file --------------------------------------------
write_cifti(xifti = unicorn.dscalar, cifti_fname = "/Users/ashfrana/Desktop/code/ABCD GAMs replication/gam_sex_abs_sum_all_networks_replication_2.dscalar.nii")
ciftiTools.setOption('wb_path', '/Users/ashfrana/Desktop/workbench')
