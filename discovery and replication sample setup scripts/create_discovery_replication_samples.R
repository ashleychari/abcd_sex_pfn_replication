data_for_ridge <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/data_for_ridge_030824.csv")


set.seed(42)
discovery_sample <- subset(data_for_ridge, matched_group==1)
discovery_sample$rel_family_id <- as.factor(discovery_sample$rel_family_id)
ind <- sapply( unique( discovery_sample$rel_family_id ) , function(x) sample( which(discovery_sample$rel_family_id==x) , 1 ) )
discovery_sample <- discovery_sample[ind, ]

write.csv(discovery_sample, "/Users/ashfrana/Desktop/code/ABCD GAMs replication/discovery_sample_removed_siblings.csv")


replication_sample <- subset(data_for_ridge, matched_group==2)
replication_sample$rel_family_id <- as.factor(replication_sample$rel_family_id)
ind <- sapply( unique( replication_sample$rel_family_id ) , function(x) sample( which(replication_sample$rel_family_id==x) , 1 ) )
replication_sample <- replication_sample[ind, ]

write.csv(replication_sample, "/Users/ashfrana/Desktop/code/ABCD GAMs replication/replication_sample_removed_siblings.csv")

discovery_sample$matched_group[1] == 1
replication_sample$matched_group[1]
