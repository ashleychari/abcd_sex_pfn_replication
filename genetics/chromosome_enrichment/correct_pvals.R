library(stats)

uncorrected_discovery_pvals <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/chromosome_enrichment/uncorrected_discovery_pvals.csv")
corrected_pvals <- p.adjust(uncorrected_discovery_pvals$pval, method="fdr")

corrected_pval_df <- as.data.frame(uncorrected_discovery_pvals$chromosome)
corrected_pval_df$corrected_pval <- corrected_pvals

write.csv("figure_4a_fdr_corrected_pvals")