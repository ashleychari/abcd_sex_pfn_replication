library(ggplot2)
library(hexbin)
library(RcppCNPy)
library(stringr)
library(sjmisc)
library(grDevices)
theme_set(theme_classic(base_size = 16))

# Plots to be made
# 1. GAMs discovery vs replication - DONE
# 2. SVM haufe discovery vs replication - DONE
# 3. GAMs discovery vs SVM haufe discovery - DONE
# 4. GAMs replication vs SVM haufe replication - DONE
# 5. PNC GAMs discovery vs ABCD GAMs discovery



args = commandArgs(trailingOnly=TRUE)
file_1 <- args[1]
data_label_1 <- args[2]
file_2 <- args[3]
data_label_2 <- args[4]
save_filename <- args[5]

print(args)

# Under assumption that svm files will by npy format and gams will be csv
load_file <- function(filename){
  if (str_contains(filename, ".csv")){
    file_data <- read.csv(filename)
    file_df <- data.frame(file_data$X0)
    colnames(file_df) <- "vals"
    file_label <- "gams"
  }
  else{
    file_data <- npyLoad(filename)
    file_df <- data.frame(file_data)
    colnames(file_df) <- "vals"
    file_label <- "svm"
  }
  
  return (list(file_df, file_label))
}



file_1_df <- load_file(file_1)[[1]]
file_1_label <- load_file(file_1)[[2]]
file_2_df <- load_file(file_2)[[1]]
file_2_label <- load_file(file_2)[[2]]

#gams_abs_sum_discovery <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/uncorrected_abs_sum_matrix_discovery.csv")
#gams_abs_sum_replication <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/uncorrected_abs_sum_matrix_replication.csv")
#svm_abs_sum_discovery <- npyLoad("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/svm_072324_run/abs_sum_weight_brain_mat_discovery_haufe_100_runs_072324.npy")


#data_brain_gam_disc_all <- data.frame(gams_abs_sum_discovery$X0)
#colnames(data_brain_gam_disc_all) <- "loadings"

#data_brain_gam_rep_all <- data.frame(gams_abs_sum_replication$X0)
#colnames(data_brain_gam_rep_all) <- "loadings"

#data_brain_svm_disc_all <- data.frame(svm_abs_sum_discovery)
#colnames(data_brain_svm_disc_all) <- "weights"


#myPalette <- grey.colors(7, start=0.4, end=0.9)
myPalette <- c("#333333", "#4C4C4C", "#666666", "#7F7F7F", "#999999", "#B2B2B2", "#CCCCCC")



Data_tmp1 = data.frame(file_1_vals = as.numeric(file_1_df$vals));
Data_tmp1$file_2_vals = as.numeric(file_2_df$vals);
cor.test(Data_tmp1$file_1_vals, Data_tmp1$file_2_vals, method = "pearson")

hexinfo <- hexbin(Data_tmp1$file_1_vals, Data_tmp1$file_2_vals, xbins = 30);
data_hex <- data.frame(hcell2xy(hexinfo), count = hexinfo@count);


# gams and gams
# svm and gams
if (file_1_label == "gams" & file_2_label == "gams"){
  p <- ggplot() +
    geom_hex(data = subset(data_hex, count > 10), aes(x, y, fill = count), stat = "identity") +
    scale_fill_gradientn(colours = myPalette, name = "Count", limit=c(0, 1200), breaks=seq(400, 1000, 300)) + 
    geom_smooth(data = Data_tmp1, aes(x = file_1_vals, y = file_2_vals), method = lm, color = "#FFFFFF", linetype = "dashed") +
    theme_classic() + labs(x = data_label_1, y = data_label_2) + 
    scale_x_continuous(breaks=seq(0, 50, 10), limits=c(0, 50)) +
    scale_y_continuous(breaks=seq(0, 50, 10), limits=c(0, 50)) +
    theme(legend.text = element_text(size = 10), legend.title = element_text(size = 15)) +
    theme(legend.justification = c(1, 1), legend.position = c(1, 1)) + theme(axis.text.x = element_text(size= 10), axis.text.y = element_text(size= 10), axis.title=element_text(size=15)) + guides(colour= guide_colorbar(barwidth=unit(0.5, "cm")))
}else if (file_1_label == "svm" & file_2_label == "gams") {
  p <- ggplot() +
    geom_hex(data = subset(data_hex, count > 10), aes(x, y, fill = count), stat = "identity") +
    scale_fill_gradientn(colours = myPalette, name = "Count", limit=c(0, 1200), breaks=seq(400, 1000, 300)) + 
    geom_smooth(data = Data_tmp1, aes(x = file_1_vals, y = file_2_vals), method = lm, color = "#FFFFFF", linetype = "dashed") +
    theme_classic() + labs(x = data_label_1, y = data_label_2) + 
    scale_x_continuous(breaks=seq(0, 0.25, 0.05), limits=c(0, 0.25)) +
    scale_y_continuous(breaks=seq(0, 50, 10), limits=c(0, 50)) +
    theme(legend.text = element_text(size = 10), legend.title = element_text(size = 15)) +
    theme(legend.justification = c(1, 1), legend.position = c(1, 1)) + theme(axis.text.x = element_text(size= 10), axis.text.y = element_text(size= 10), axis.title=element_text(size=15)) + guides(colour= guide_colorbar(barwidth=unit(0.5, "cm")))
}else if (file_1_label == "gams" & file_2_label == "svm"){
  p <- ggplot() +
    geom_hex(data = subset(data_hex, count > 10), aes(x, y, fill = count), stat = "identity") +
    scale_fill_gradientn(colours = myPalette, name = "Count", limit=c(0, 1200), breaks=seq(400, 1000, 300)) + 
    geom_smooth(data = Data_tmp1, aes(x = file_1_vals, y = file_2_vals), method = lm, color = "#FFFFFF", linetype = "dashed") +
    theme_classic() + labs(x = data_label_1, y = data_label_2) + 
    scale_x_continuous(breaks=seq(0, 50, 10), limits=c(0, 50)) +
    scale_y_continuous(breaks=seq(0, 0.25, 0.05), limits=c(0, 0.25)) +
    theme(legend.text = element_text(size = 10), legend.title = element_text(size = 15)) +
    theme(legend.justification = c(1, 1), legend.position = c(1, 1)) + theme(axis.text.x = element_text(size= 10), axis.text.y = element_text(size= 10), axis.title=element_text(size=15)) + guides(colour= guide_colorbar(barwidth=unit(0.5, "cm")))
}else{ # svm vs svm
  p <- ggplot() +
    geom_hex(data = subset(data_hex, count > 10), aes(x, y, fill = count), stat = "identity") +
    scale_fill_gradientn(colours = myPalette, name = "Count", limit=c(0, 1200), breaks=seq(400, 1000, 300)) + 
    geom_smooth(data = Data_tmp1, aes(x = file_1_vals, y = file_2_vals), method = lm, color = "#FFFFFF", linetype = "dashed") +
    theme_classic() + labs(x = data_label_1, y = data_label_2) + 
    scale_x_continuous(breaks=seq(0, 0.25, 0.05), limits=c(0, 0.25)) +
    scale_y_continuous(breaks=seq(0, 0.25, 0.05), limits=c(0, 0.25)) +
    theme(legend.text = element_text(size = 10), legend.title = element_text(size = 15)) +
    theme(legend.justification = c(1, 1), legend.position = c(1, 1)) + theme(axis.text.x = element_text(size= 10), axis.text.y = element_text(size= 10), axis.title=element_text(size=15)) + guides(colour= guide_colorbar(barwidth=unit(0.5, "cm")))
}

print(save_filename)
ggsave(save_filename, plot=p, height=3.5, width=3.5, dpi=300)
