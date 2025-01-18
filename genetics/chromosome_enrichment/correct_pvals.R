library(stats)

uncorrected_discovery_pvals <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/chromosome_enrichment/uncorrected_discovery_pvals.csv")
corrected_pvals <- p.adjust(uncorrected_discovery_pvals$pval, method="bonferroni")

corrected_pval_df <- as.data.frame(uncorrected_discovery_pvals$chromosome)
corrected_pval_df$uncorrected_pval <- uncorrected_discovery_pvals$pval
corrected_pval_df$corrected_pval <- corrected_pvals

write.csv(corrected_pval_df, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/chromosome_enrichment/figure_4a_fdr_corrected_pvals_2.csv")


uncorrected_rep_pvals <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/chromosome_enrichment/uncorrected_replication_pvals.csv")
corrected_rep_pvals <- p.adjust(uncorrected_rep_pvals$pval, method="bonferroni")

corrected_rep_pval_df <- as.data.frame(uncorrected_rep_pvals$chromosome)
corrected_rep_pval_df$uncorrected_pval <- uncorrected_rep_pvals$pval
corrected_rep_pval_df$corrected_pval <- corrected_rep_pvals

write.csv(corrected_rep_pval_df, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/chromosome_enrichment/replication_corrected_pvals_bonf.csv")
