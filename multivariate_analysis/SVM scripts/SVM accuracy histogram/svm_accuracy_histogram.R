library(ggplot2)
accuracies <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/svm_correct_results/permutation_accuracy_table.csv")

ggplot(accuracies, aes(x=discovery_accuracies)) + geom_histogram(bins=200) + geom_vline(aes(xintercept = as.numeric(0.92)),
                                                                                    color = "red", linetype = "dashed") + labs(title='Discovery permutation accuracies', x="Accuracy") +
  scale_y_continuous(name="Count", expand = c(0, 0)) +  
  scale_x_continuous(breaks=c(0.4, 0.6, 0.8), labels=c("0.4", "0.6", "0.8")) +
  theme(axis.text.y = element_text (family="Arial", size = 10)) +
  theme(axis.title = element_text (family="Arial", size = 12, colour= "black")) +
  theme(axis.text.x = element_text (family="Arial", size = 10)) 

ggplot(accuracies, aes(x=replication_accuracies)) + geom_histogram(bins=200) + geom_vline(aes(xintercept = as.numeric(0.91)),
                                                                                    color = "red", linetype = "dashed") + labs(title='Replication permutation accuracies', x="Accuracy") +
  scale_y_continuous(name="Count", expand = c(0, 0)) +  
  scale_x_continuous(breaks=c(0.4, 0.6, 0.8), labels=c("0.4", "0.6", "0.8")) +
  theme(axis.text.y = element_text (family="Arial", size = 10)) +
  theme(axis.title = element_text (family="Arial", size = 12, colour= "black")) +
  theme(axis.text.x = element_text (family="Arial", size = 10)) 