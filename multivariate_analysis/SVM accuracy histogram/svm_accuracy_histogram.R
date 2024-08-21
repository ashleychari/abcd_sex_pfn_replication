library(ggplot2)
discovery_accuracies <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/SVM accuracy histogram/discovery_permutation_accuracy_081524.csv")
replication_accuracies <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/SVM accuracy histogram/replication_permutation_accuracy_081524.csv")

disc_hist <- ggplot(discovery_accuracies, aes(x=Accuracy)) + geom_histogram(bins=200) + geom_vline(aes(xintercept = as.numeric(0.871)),
                                                                                    color = "red", linetype = "dashed") + labs(x="Accuracy") +
  scale_y_continuous(name="Count", expand = c(0, 0)) +  
  scale_x_continuous(breaks=c(0.4, 0.6, 0.8), labels=c("0.4", "0.6", "0.8")) +
  theme(axis.text.y = element_text (family="Arial", size = 20)) +
  theme(axis.title = element_text (family="Arial", size = 20, colour= "black")) +
  theme(axis.text.x = element_text (family="Arial", size = 20)) 


ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/SVM accuracy histogram/discovery_permuation_histogram.png", 
       disc_hist, dpi=500)


rep_hist <- ggplot(replication_accuracies, aes(x=Accuracy)) + geom_histogram(bins=200) + geom_vline(aes(xintercept = as.numeric(0.874)),
                                                                                    color = "red", linetype = "dashed") + labs(x="Accuracy") +
  scale_y_continuous(name="Count", expand = c(0, 0)) +  
  scale_x_continuous(breaks=c(0.4, 0.6, 0.8), labels=c("0.4", "0.6", "0.8")) +
  theme(axis.text.y = element_text (family="Arial", size = 20)) +
  theme(axis.title = element_text (family="Arial", size = 20, colour= "black")) +
  theme(axis.text.x = element_text (family="Arial", size = 20)) 

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/SVM accuracy histogram/replication_permuation_histogram.png", 
       rep_hist, dpi=500)