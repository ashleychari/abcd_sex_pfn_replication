library(ggplot2)

discovery_network_accuracies <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/svm_check_run/discovery_network_specific_model_accuracies.csv")
replication_network_accuracies <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/svm_check_run/replication_network_specific_model_accuracies.csv")

netColor <- c("#E76178", "#7499C2", "#F5BA2E", "#7499C2", "#00A131",  "#AF33AD", "#E443FF","#E76178"
              ,"#E443FF", "#AF33AD",  "#7499C2", "#E76178","#7499C2","#00A131", "#F5BA2E",
              "#4E31A8", "#F5BA2E")

discovery_network_accuracies <- cbind(discovery_network_accuracies, netColor)
discovery_ordered <- discovery_network_accuracies[order(discovery_network_accuracies$Accuracy, decreasing=FALSE), ]
discovery_network_accuracies$netColor <- factor(discovery_network_accuracies$netColor)


disc_plot <- ggplot(discovery_network_accuracies, aes(x=reorder(Network, +Accuracy), y=Accuracy, fill=netColor)) + geom_bar(stat='identity') + 
  scale_fill_identity() + theme(legend.position="none") + theme(axis.text.x = element_text(size= 12), axis.text.y = element_text(size= 12), axis.title = element_text(size=18)) +
  xlab("Network") + ylab("Model Performance") + coord_cartesian(ylim = c(0.55, 0.75)) + scale_y_continuous(breaks=seq(0.55, 0.75, 0.05))

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/network_specific_analyses/plots/discovery_network_specific_analyses.png", disc_plot, width=4.8, height=3.5, dpi=300)


disc_plot


replication_network_accuracies <- cbind(replication_network_accuracies, netColor)
replication_ordered <- replication_network_accuracies[order(replication_network_accuracies$Accuracy, decreasing=FALSE), ]
replication_network_accuracies$netColor <- factor(replication_network_accuracies$netColor)


rep_plot <- ggplot(replication_network_accuracies, aes(x=reorder(Network, +Accuracy), y=Accuracy, fill=netColor)) + geom_bar(stat='identity') + 
  scale_fill_identity() + theme(legend.position="none") + theme(axis.text.x = element_text(size= 12), axis.text.y = element_text(size= 12), axis.title = element_text(size=18)) +
  xlab("Network") + ylab("Model Performance") + coord_cartesian(ylim = c(0.55, 0.75)) + scale_y_continuous(breaks=seq(0.55, 0.75, 0.05))

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/network_specific_analyses/plots/replication_network_specific_analyses.png", rep_plot, width=4.8, height=3.5, dpi=300)

rep_plot