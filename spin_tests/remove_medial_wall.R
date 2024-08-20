library(ciftiTools)
library(gifti)


pnc_gams_fslr_abs_sum <- readgii("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/Gam_abs_sum_fslr.gii")

pnc_gams_data <- pnc_gams_fslr_abs_sum$data$unknown

cifti_lh2 <- as.numeric(pnc_gams_data)[1:32492]
cifti_rh2 <- as.numeric(pnc_gams_data)[32493:64984]

medialwall.mask.leftcortex <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.leftcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask
medialwall.mask.rightcortex <-read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/medialwall.mask.rightcortex.csv", header = F, colClasses = c("logical")) #left cortex medial wall mask

original_map_cifti <- as_cifti(cortexL = cifti_lh2, cortexL_mwall = medialwall.mask.leftcortex$V1, cortexR =  cifti_rh2, cortexR_mwall = medialwall.mask.rightcortex$V1)

pnc_gams_no_medial_wall <- move_to_mwall(original_map_cifti)

pnc_gams_no_medial_wall_all <- data.frame(X0 = c(pnc_gams_no_medial_wall$data$cortex_left, pnc_gams_no_medial_wall$data$cortex_right))


write.csv(pnc_gams_no_medial_wall_all, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/spin_tests/data/PNC_data/PNC_gams_abs_sum_all_fslr.csv")
