library(pROC)
library(plotROC)
theme_set(theme_classic(base_size = 16))

svm_discovery_data <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/discovery_svm_ROC_siblings_removed_data.csv")
svm_replication_data <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/replication_svm_ROC_siblings_removed_data.csv")

svm_disc_roc <- roc(svm_discovery_data$sex ~ svm_discovery_data$decision_values)
svm_rep_roc <- roc(svm_replication_data$sex ~ svm_replication_data$decision_values)


svm_disc_rocplot <- ggplot(svm_discovery_data, aes(m = decision_values, d = sex)) + 
  geom_roc(n.cuts=0, labels=FALSE, size=2) + xlab("1 - Specificity") + ylab("Sensitivity") +
  scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2), labels=c(0, 0.2, 0.4, 0.6, 0.8, 1) , expand=expansion(0)) + scale_x_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2), labels=c(0, 0.2, 0.4, 0.6, 0.8, 1), expand=expansion(0)) +
  theme(axis.text.x=element_text(size=15), axis.text.y=element_text(size=15), axis.title=element_text(size=20))

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/multivariate_analysis/roc_curves/svm_discovery_roc_plot.png", 
       svm_disc_rocplot, width=4.8, height=3.5, dpi=300)

svm_rep_rocplot <- ggplot(svm_replication_data, aes(m = decision_values, d = sex)) + 
  geom_roc(n.cuts=0, labels=FALSE, size=2) + xlab("1 - Specificity") + ylab("Sensitivity") +
  scale_y_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2), labels=c(0, 0.2, 0.4, 0.6, 0.8, 1) , expand=expansion(0)) + scale_x_continuous(limits=c(0, 1), breaks=seq(0, 1, 0.2), labels=c(0, 0.2, 0.4, 0.6, 0.8, 1), expand=expansion(0)) +
  theme(axis.text.x=element_text(size=15), axis.text.y=element_text(size=15), axis.title=element_text(size=20))

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/multivariate_analysis/roc_curves/svm_replication_roc_plot.png", 
       svm_rep_rocplot, width =4.8, height=3.5, dpi=300)

