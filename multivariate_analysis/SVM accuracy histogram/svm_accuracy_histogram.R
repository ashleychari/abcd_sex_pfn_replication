library(ggplot2)
theme_set(theme_classic(base_size = 16))
discovery_accuracies <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/SVM accuracy histogram/discovery_permutation_accuracy_081524.csv")
replication_accuracies <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/SVM accuracy histogram/replication_permutation_accuracy_081524.csv")


ggplot(discovery_accuracies, aes(x=Accuracy)) + geom_histogram(fill="black")


disc_hist <- ggplot(discovery_accuracies, aes(x=Accuracy)) + geom_histogram(fill="black", bins=200, size=0.02, color="black") + geom_vline(aes(xintercept = as.numeric(0.874)),
                                                                                    color = "red", linetype = "dashed", linewidth=1) + labs(x="Accuracy", y="Count") +
  scale_y_continuous(limits = c(0, 80), breaks=seq(0, 80, 20), expand = c(0, 0)) +  
  scale_x_continuous(breaks=c(0.4, 0.6, 0.8), labels=c("0.4", "0.6", "0.8")) +
  theme(axis.text.y = element_text (size = 7)) +
  theme(axis.title = element_text (size = 10, colour= "black")) +
  theme(axis.text.x = element_text (size = 7)) + theme(axis.line=element_line(linewidth=1))

disc_hist


ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/multivariate_analysis/permutation_histograms/discovery_permuation_histogram.png", 
       disc_hist, width=2, height=2, dpi=500)


rep_hist <- ggplot(replication_accuracies, aes(x=Accuracy)) + geom_histogram(color="black", bins=200, size=0.02, fill="black") + geom_vline(aes(xintercept = as.numeric(0.871)),
                                                                                    color = "red", linetype = "dashed", linewidth=1) + labs(x="Accuracy", "Count") +
  scale_y_continuous(name="Count", limits = c(0, 80), breaks=seq(0, 80, 20), expand = c(0, 0)) +  
  scale_x_continuous(breaks=c(0.4, 0.6, 0.8), labels=c("0.4", "0.6", "0.8")) +
  theme(axis.text.y = element_text (size = 7)) +
  theme(axis.title = element_text (size = 10, colour= "black")) +
  theme(axis.text.x = element_text (size = 7)) + theme(axis.line=element_line(linewidth=1))

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/multivariate_analysis/permutation_histograms/replication_permuation_histogram.png", 
       rep_hist, width=2, height=2, dpi=500)
