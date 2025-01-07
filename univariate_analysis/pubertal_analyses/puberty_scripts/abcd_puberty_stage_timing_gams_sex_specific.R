library(mgcv)
library(RcppCNPy)
library(R.matlab)
library(lme4)

set.seed(42)

# sets up the behavior new discovery dataframe
create_behavior_df <- function(filepath, sex){
  behavior_df <- read.csv(filepath)
  behavior_new <- data.frame(subjectkey=behavior_df$subjectkey)
  behavior_new$AgeYears <- as.numeric(behavior_df$interview_age/12);
  behavior_new$Motion <- as.numeric(behavior_df$meanFD);
  behavior_new$Sex_factor <- as.factor(behavior_df$sex);
  behavior_new$sex01 <- ifelse(behavior_new$Sex_factor=="M", 0,1)
  behavior_new$oSex <- ordered(behavior_new$sex01,levels=c("0","1"))
  behavior_new$matched_group <- as.factor(behavior_df$matched_group);
  behavior_new$abcd_site <- as.factor(behavior_df$abcd_site)
  behavior_new$rel_family_id <- as.factor(behavior_df$rel_family_id)
  if (sex == "F"){
    behavior_new$pds_p_ss_female_category <- as.factor(behavior_df$pds_p_ss_female_category)
  }else{
    behavior_new$pds_p_ss_male_category <- as.factor(behavior_df$pds_p_ss_male_category)
  }
  return (behavior_new)
}

# Applies inputted correction method to inputted pvector
fdr_correction <- function(p_vec, z_vec, correction_type){
    p_fdr_vec <- p.adjust(p_vec, method=correction_type)
    z_sig_vec <- z_vec
    # if p value is greater than 0.05 replace with 0 for the z vector
    z_sig_vec[which(p_fdr_vec >= 0.05)] <- 0
    return(list(p_fdr=p_fdr_vec, z_sig=z_sig_vec))
}

# Creates a result vector to hold info in 
create_results_vec <- function(vec_to_save, nonzero_column_index, features_quantity){
    result_vec <- matrix(0, 1, features_quantity)
    result_vec[nonzero_column_index] <- vec_to_save
    return (result_vec)
}

main <- function(behavior_filename, set, test_name, include_age, sex){
    RESULTS_FOLDER <- paste0("/cbica/projects/ash_pfn_sex_diff_abcd/results/puberty_analyses_2/", set, '/', test_name)
    # NEED BEHAVIOR NEW DISCOVERY FILE READ IN (ask arielle)
    behavior_new <- create_behavior_df(behavior_filename, sex)

    # Setup for matrices
    subjects_quantity <- nrow(behavior_new) # data for ridge subset has this many patients, 3240
    features_quantity <- 59412
    data_size <- subjects_quantity * features_quantity
    subject_keys <- behavior_new$subjectkey

    # for each of the 17 networks
    for (i in 1:17){
        ######## NEED TO COMMENT THIS CODE #############
        Data_I <- matrix(0,1, data_size);
        dim(Data_I) <- c(subjects_quantity, features_quantity);
        # for each subject, add data to matrix Data_I (each row is a subject?)
        for (k in 1:length(subject_keys)){
            atlas_file <- paste0("/cbica/projects/ash_pfn_sex_diff_abcd/results/converted_matrices/sub-NDAR", 
            as.character(subject_keys[k]), "/parcel_1by1/converted_UV_mat.csv")
            mat <- read.csv(atlas_file)
            # insert row into loadings matrix for each of the subjects for each of the networks
            # Data_I will contain the vertices for each of the networks for each of the subjects?
            # One network at a time
            Data_I[k,] <- mat[1:59412, i] 
        }
        # Get nonzero columns
        ColumnSum = colSums(Data_I);
        NonZero_ColumnIndex = which(ColumnSum != 0);
        # Subset dataframe for nonzero columns
        Data_I_NonZero = Data_I[, NonZero_ColumnIndex];
        # Get nonzero column count
        FeaturesQuantity_NonZero = dim(Data_I_NonZero)[2];
        ###################################################

        gam_P_Vector <- matrix(0, 1, FeaturesQuantity_NonZero);
        gam_Z_Vector <- matrix(0, 1, FeaturesQuantity_NonZero);
        gam_P_FDR_Vector <- matrix(0, 1, FeaturesQuantity_NonZero);
        gam_P_Bonf_Vector <- matrix(0, 1, FeaturesQuantity_NonZero);
        
        # Gam analysis for all of the non zero features in the df
        for (j in 1:FeaturesQuantity_NonZero){
            # subset Data_I_NonZero for column (<59412 columns); essentially loading is
            # weights for each vertex for each participant
            loading_data <- as.numeric(Data_I_NonZero[,j]) 
            # gam analysis with random effects for rel_family_id and abcd_site
            if (include_age == "TRUE" & sex == 'F'){
                print("Running gam_analysis <- bam(loading_data ~ pds_p_ss_female_category + Motion + 
                s(AgeYears, k=4) + s(abcd_site, bs='re', k=4), method = fREML, data = behavior_new)")
                gam_analysis <- bam(loading_data ~  pds_p_ss_female_category + Motion + s(AgeYears, k=4) + s(abcd_site, bs='re', k=4), method = "fREML", data = behavior_new);
                lm_analysis <- lmer(loading_data ~   pds_p_ss_female_category + Motion + AgeYears + (1|abcd_site), data = behavior_new);
            }else if (include_age == "FALSE" & sex == 'F'){
                print("Running gam_analysis <- bam(loading_data ~  pds_p_ss_female_category + Motion  +
                 s(abcd_site, bs='re', k=4), method = fREML, data = behavior_new)")
                gam_analysis <- bam(loading_data ~ pds_p_ss_female_category + Motion + s(abcd_site, bs='re', k=4), method = "fREML", data = behavior_new);
                lm_analysis <- lmer(loading_data ~ pds_p_ss_female_category + Motion + (1|abcd_site), data = behavior_new);
            }else if (include_age == "TRUE" & sex == 'M'){
                print("Running gam_analysis <- bam(loading_data ~ pds_p_ss_male_category + Motion + 
                s(AgeYears, k=4) + s(abcd_site, bs='re', k=4), method = fREML, data = behavior_new)")
                gam_analysis <- bam(loading_data ~ pds_p_ss_male_category + Motion + s(AgeYears, k=4) + s(abcd_site, bs='re', k=4), method = "fREML", data = behavior_new);
                lm_analysis <- lmer(loading_data ~ pds_p_ss_male_category + Motion + AgeYears + (1|abcd_site), data = behavior_new);
            }else{
                print("Running gam_analysis <- bam(loading_data ~  pds_p_ss_male_category + Motion
                + s(abcd_site, bs='re', k=4), method = fREML, data = behavior_new)")
                gam_analysis <- bam(loading_data ~  pds_p_ss_male_category + Motion + s(abcd_site, bs='re', k=4), method = "fREML", data = behavior_new);
                lm_analysis <- lmer(loading_data ~ pds_p_ss_male_category + Motion +(1|abcd_site), data = behavior_new);
            }
            
            gam_P_Vector[j] <- summary(gam_analysis)$p.pv[2];
            gam_Z_Vector[j] <- qnorm(gam_P_Vector[j] / 2, lower.tail = FALSE);
            
            pds_T <- summary(lm_analysis)$coefficients[2, 3];
            if (pds_T < 0) {
                gam_Z_Vector[j] <- -gam_Z_Vector[j];
            }
        }

        if (behavior_new$matched_group[1] == 1){
            sex_effect_label <-  "/SexEffect_AtlasLoading_Discovery_17_Network_"
        }
        else{
            sex_effect_label <- "/SexEffect_AtlasLoading_Replication_17_Network_"
        }

        # FDR correction
        fdr_correction_results <- fdr_correction(gam_P_Vector, gam_Z_Vector, "fdr")
        gam_p_fdr_vector <- fdr_correction_results$p_fdr
        gam_z_fdr_sig_vector <- fdr_correction_results$z_sig
        bonferroni_correction_results <- fdr_correction(gam_P_Vector, gam_Z_Vector, "bonferroni")
        gam_p_bonf_vector <- bonferroni_correction_results$p_fdr
        gam_z_bonf_sig_vector <- bonferroni_correction_results$z_sig
        
        # Write results into matrix
        # Saves all nonzero columns into one long vector of full length features_quantitiy 
        gam_p_vector_all <- create_results_vec(gam_P_Vector, NonZero_ColumnIndex, features_quantity)
        gam_z_vector_all <- create_results_vec(gam_Z_Vector, NonZero_ColumnIndex, features_quantity)
        gam_p_fdr_vector_all <- create_results_vec(gam_p_fdr_vector, NonZero_ColumnIndex, features_quantity)
        gam_z_fdr_vector_sig_all <- create_results_vec(gam_z_fdr_sig_vector, NonZero_ColumnIndex, features_quantity)
        gam_p_bonf_vector_all <- create_results_vec(gam_p_bonf_vector, NonZero_ColumnIndex, features_quantity)
        gam_z_bonf_vector_sig_all <- create_results_vec(gam_z_bonf_sig_vector, NonZero_ColumnIndex, features_quantity)
        
        writeMat(paste0(RESULTS_FOLDER, sex_effect_label, as.character(i), '.mat'), 
        Gam_P_Vector_All = gam_p_vector_all, Gam_Z_Vector_All = gam_z_vector_all, Gam_P_FDR_Vector_All = gam_p_fdr_vector_all,
        Gam_Z_FDR_Sig_Vector_All = gam_z_fdr_vector_sig_all, Gam_P_Bonf_Vector_All = gam_p_bonf_vector_all, 
        Gam_Z_Bonf_Sig_Vector_All = gam_z_bonf_vector_sig_all)
        

        results_df <- data.frame(Gam_P_Vector_All = c(gam_p_vector_all), Gam_Z_Vector_All = c(gam_z_vector_all), Gam_P_FDR_Vector_All = c(gam_p_fdr_vector_all),
        Gam_Z_FDR_Sig_Vector_All = c(gam_z_fdr_vector_sig_all), Gam_P_Bonf_Vector_All = c(gam_p_bonf_vector_all), 
        Gam_Z_Bonf_Sig_Vector_All = c(gam_z_bonf_vector_sig_all))
        write.csv(results_df, paste0(RESULTS_FOLDER, sex_effect_label, as.character(i), '.csv'))
    } 
}

args = commandArgs(trailingOnly=TRUE)
print("IN abcd_puberty_stage_timing_gams_sex_specific.R")
set <- args[1]
behavior_file <- args[2]
test_name <- args[3]
include_age <- args[4]
sex <- args[5]
print(paste0("set: ", set))
print(paste0("behavior file: ", behavior_file))

# Run script
main(behavior_file, set, test_name, include_age, sex) # discovery sample
#main(2) # replication sample