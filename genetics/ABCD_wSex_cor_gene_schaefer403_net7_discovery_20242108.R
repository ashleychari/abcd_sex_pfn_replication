
library(R.matlab)
library(dplyr)
library(magrittr)
library(tibble)
library(ggplot2)
library(data.table)
library(ggplot2)
library(hexbin)
library(matrixStats)
library(plotrix)
library(scales)
library(gifti)
theme_set(theme_classic(base_size = 16))


#brain input is from SVM run 100x
# read in LH parc
lh_schaefer1000 <- read.table("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/lh_Schaefer1000_7net_L.csv")
lh_schaefer1000 <- lh_schaefer1000[1:10242,]
# read in color lookup table
lh_schaefer1000_ct <- read.table("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/lh_Schaefer1000_7net_cttable.csv")
lh_schaefer1000_ct_500 <- as.data.frame(lh_schaefer1000_ct[-1,])
colnames(lh_schaefer1000_ct_500) <- "V1"
setDT(lh_schaefer1000_ct_500, keep.rownames = "parcNum")

# read in brain img data
#data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_uncorrected_discovery_LH.fsaverage5.func.gii')
#data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_uncorrected_replication_LH.fsaverage5.func.gii')
#data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_fdr_discovery_LH.fsaverage5.func.gii')
data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_fdr_replication_LH.fsaverage5.func.gii')
data_brain <- data_brain1$data$normal


# parcellate data
data_vec <- as.vector(data_brain)
data_parc_lh <- cbind(data_vec, lh_schaefer1000) %>% as_tibble() %>% group_by(lh_schaefer1000) %>% summarise(wt_mean = mean(data_vec))

#order by parcel number for rotations
data_parc_lh <- subset(data_parc_lh, lh_schaefer1000 != "65793")
data_parc_lh_parcnum <- data_parc_lh
data_parc_lh_parcnum$parcNum <- "NA"

for (i in 1:500){
  for (j in 1:500){
    if (data_parc_lh_parcnum$lh_schaefer1000[i] == lh_schaefer1000_ct_500$V1[j]){
      data_parc_lh_parcnum$parcNum[i] <- lh_schaefer1000_ct_500$parcNum[j] }}}

data_parc_lh_parcnum$parcNum <- as.numeric(data_parc_lh_parcnum$parcNum)
data_parc_lh_parcnum_sort <- data_parc_lh_parcnum[order(data_parc_lh_parcnum$parcNum),]


###read in gene data
geneinfo <- readMat("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/AHBAprocessed/ROIxGene_Schaefer1000_INT.mat")
geneinfo1 <- lapply(geneinfo, unlist, use.names=FALSE)
geneMat <- as.data.frame(geneinfo1$parcelExpression)
geneMat1 <- subset(geneMat, select = -V1)
gene <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/GeneSymbol.csv",header=TRUE)
colnames(gene) <- "V1"
colnames(geneMat1) <- gene$V1
geneMat2 <- cbind(lh_schaefer1000_ct_500, geneMat1)
geneMat2$parcNum <- as.numeric(geneMat2$parcNum)
geneMat2_sort <- geneMat2[order(geneMat2$parcNum),]
geneMat2_sort_500 <- subset(geneMat2_sort, select=-V1)


#merge img, gene, and centroid coord. remove parcels with missing gene data
img_geneMatChrom_500 <- merge(data_parc_lh_parcnum_sort,geneMat2_sort_500, by="parcNum")


rois <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics_files/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mmCentroid_RAS.csv") 
#above csv downloaded from here- https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv
#removed spaces in column names prior to reading in
roislh <- rois[1:500,]

img_geneMatChrom_roi_500 <- merge(roislh, img_geneMatChrom_500, by.x = "RoiLabel", by.y= "parcNum",all = FALSE)
img_geneMatChrom_roi_500_comp <- subset(img_geneMatChrom_roi_500, !is.nan(img_geneMatChrom_roi_500[[10]])) ##if add more columns, change "10". This is the complete DF. Any subsets below should come from here/match this order.


data_parc_lh_sort_500_comp <- subset(img_geneMatChrom_roi_500_comp, select=wt_mean)


othercols <- c(colnames(rois), "lh_schaefer1000", "wt_mean")
geneMatChrom_500_comp <- img_geneMatChrom_roi_500_comp[(length(othercols)+1):(length(colnames(img_geneMatChrom_roi_500_comp)))]




#Select genes for chromosome enrichement 
chrom <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics_files/Richiardi_Data_File_S2_ChrAdded.csv")
chromRefined <- chrom$chromosome_name[chrom$gene_symbol %in% gene$V1] #16000, which of those are in the 20, 000? indexing the 16000
geneRefined <- chrom$gene_symbol[chrom$gene_symbol %in% gene$V1] #16000, which of those are in the 20, 000? indexing the 16000
geneMatChrom_500_comp_fin <-geneMatChrom_500_comp[,match(geneRefined,colnames(geneMatChrom_500_comp),nomatch = 0)] #target gene first. where in #2 is #1




geneRefined_df <- as.data.frame(geneRefined)
chromRefined_df <- as.data.frame(chromRefined)
chromGeneRefined <- cbind(chromRefined_df, geneRefined_df)


chromGeneRefined$geneRefined <- gsub("-", "." , chromGeneRefined$geneRefined)
t2 <- sapply(1:12986, function(x) cor(data_parc_lh_sort_500_comp,geneMatChrom_500_comp_fin[,x]))
colt2<- as.vector(chromGeneRefined$geneRefined)
dft2 <- as.data.frame(cbind(colt2, t2))
colnames(dft2)<- c("gene", "corr")
df3 <- merge(dft2, chromGeneRefined, by.x="gene", by.y="geneRefined", all=TRUE)

df3$corrSigRanked <- rank(df3$corr) #rank genes based on correlation with SVM weights

#pvalues for classical method, using df3
df3$chromRefined <- as.factor(df3$chromRefined)
t3 <- sapply(levels(df3$chromRefined)[-1], function(x) median(df3$corrSigRanked[which(df3$chromRefined==x)],na.rm = TRUE)) 
t4<- t3 - median(df3$corrSigRanked) # for the average correlation ranking for each chromosome, how far away it is from the median
t4
t5 <- as.data.frame(t4)
setDT(t5, keep.rownames = "chromosome")
t5$chromosome <- as.factor(t5$chromosome)

t5$t7 <- ifelse(is.na(t5$t4), "0", t5$t4)
t5$t7 <- as.numeric(t5$t7)


perm.pval <- data.frame(matrix(NA,nrow=length(t5$chromosome),ncol=3))
colnames(perm.pval)<- c("chromosome", "NumGenes", "pval")

t5vec <- as.list(as.character(t5$chromosome))

for (j in 1:length(t5$chromosome)){
  chromosome_tmp <- t5vec[j]
  All_tmp <- subset(df3, chromRefined == chromosome_tmp)
  genenum <- length(All_tmp$gene)
  
  if (genenum > 0) {
    null.dist.medians <-data.frame(matrix(NA,nrow=1000,ncol=1))
    colnames(null.dist.medians)<- "median_rank"
    for (i in 1:1000){
      set.seed(i)
      genelisttmp <- sample(df3$gene, genenum, replace = FALSE, prob = NULL)
      dfgenestmp <- filter(df3, gene %in% genelisttmp)
      median_rank <- median(dfgenestmp$corrSigRanked)
      null.dist.medians[i, 1] <- median_rank
    }
  } 
  newdata <- subset(t5, chromosome == chromosome_tmp)
  if (newdata$t7 > 0){
    pval<- (sum(null.dist.medians$median_rank > median(All_tmp$corrSigRanked)))/1000
  }
  if (newdata$t7 < 0){
    pval<- (sum(null.dist.medians$median_rank < median(All_tmp$corrSigRanked)))/1000
  }
  perm.pval[j, 1]  <- chromosome_tmp
  perm.pval[j, 2] <- genenum
  perm.pval[j, 3] <- pval
}


df_enrich_pval <- merge(t5, perm.pval, by = "chromosome")
df_enrich_pval1 <- df_enrich_pval[with(df_enrich_pval, order(-t7)),]
write.csv(df_enrich_pval1, '/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/chromosome_enrichment/fdr_replication_pvals.csv')




#plot for classical method, using df3
t3 <- sapply(levels(df3$chromRefined)[-1], function(x) median(df3$corrSigRanked[which(df3$chromRefined==x)],na.rm = TRUE)) # which(chromRefined==x) --  get an index of all the trues,,,   indexing out of t2Ranked where chromRefined==x.... find the median of t2Ranked for each of these levels
t4<- t3 - median(df3$corrSigRanked) # for the average correlation ranking for each chromosome, how far away it is from the median
t4
t5 <- as.data.frame(t4)
setDT(t5, keep.rownames = "chromosome")
t5$chromosome <- as.factor(t5$chromosome)

t3se <- sapply(levels(df3$chromRefined)[-1], function(x) std.error(df3$corrSigRanked[which(df3$chromRefined==x)],na.rm = TRUE)) # which(chromRefined==x) --  get an index of all the trues,,,   indexing out of t2Ranked where chromRefined==x.... find the median of t2Ranked for each of these levels
t3se
t5se <- as.data.frame(t3se)
setDT(t5se, keep.rownames = "chromosome")
t5se$chromosome <- as.factor(t5$chromosome)
t8 <- merge (t5, t5se, by = "chromosome")
colnames(t8)<- c("chromosome", "Rank", "se")

fillcolor <- c("#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", 
                        "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF",
                        "#FFFFFF", "#FFFFFF", "#FFFFFF", "grey39", "#FFFFFF",
                        "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF",
                        "#FFFFFF", "#FFFFFF", "#FF0000", "#0000FF");
                        

BorderColor <- c("grey39", "grey39", "grey39", "grey39", "grey39", 
                         "grey39", "grey39", "grey39", "grey39", "grey39",
                         "grey39", "grey39", "grey39", "grey39", "grey39",
                         "grey39", "grey39", "grey39", "grey39", "grey39",
                         "grey39", "grey39", "#FF0000", "#0000FF");
                         
LineType <- c("dashed",  "dashed", "dashed", "dashed", "dashed",
              "dashed", "dashed", "dashed", "dashed", "dashed",
              "dashed", "dashed", "dashed", "solid", "dashed",
              "dashed", "dashed", "dashed", "dashed", "dashed",
              "dashed", "dashed", "solid", "solid");

t9 <- cbind(t8, fillcolor, BorderColor, LineType)


#make color label an ordered factor so ggplot can match the color to network
t9$fillcolor <- factor(t9$fillcolor, ordered =T)
t9$BorderColor <- factor(t9$BorderColor, ordered =T)
t9$LineType <- factor(t9$LineType, ordered =T)

t9 <- t9 %>% 
  mutate(chromosome = as.factor(chromosome),fillcolor=as.character(fillcolor))%>%
  mutate(chromosome = as.factor(chromosome),BorderColor=as.character(BorderColor))%>%
  mutate(chromosome = as.factor(chromosome),LineType=as.character(LineType))%>%
  mutate(chromosome = reorder(chromosome,Rank))%>%
  arrange (chromosome)
colormap=t9 %>% select(chromosome,fillcolor)%>%unique()
linemap=t9 %>% select(chromosome,LineType)%>%unique()
bordermap=t9 %>% select(chromosome,BorderColor)%>%unique()

overall_se <- std.error(df3$corrSigRanked)


#figure 5
#tiff("/Users/sheilash/Desktop/projects/pfn_sex_diff/paper/figures/genetics/chrom_enrichments.tiff", width = 5, height = 5, units = 'in', res = 300)
p<-ggplot(data=t9, aes(x=chromosome, y=Rank)) +
  geom_bar(stat="identity", fill=colormap$fillcolor ,
           colour = bordermap$BorderColor, linetype = linemap$LineType, width = 0.8) + geom_errorbar(aes(ymin=Rank-se, ymax=Rank+se), width=0.15, size= 0.4, position=position_dodge(.9)) +
  geom_point(data=t9, aes(y=Rank, x=chromosome), size = 0.4) +
  coord_flip() + xlab("Chromosome") + ylab("Enrichment") +
  geom_hline(aes(yintercept = as.numeric(0))) +
  #ylim(-2799, 1010) +
  #scale_y_discrete(breaks = c(-2000, -1000, 1000)) +
  theme(legend.position="none") + theme(axis.text.x = element_text(size= 10), axis.text.y = element_text(size= 10, color = bordermap$BorderColor), axis.title=element_text(size = 12))
p

#dev.off()

# order by corrSigRanked
df3_gene_X <- subset(df3, df3$chromRefined=='X')
df3_gene_ranked_X <- df3_gene_X[order(df3_gene_X$corr, decreasing=TRUE), ]
df3_gene_X_ranked_top20 <- df3_gene_ranked_X[1:20, ]
write.csv(df3_gene_X_ranked_top20, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/top20_enrichment_tables/chrom_X_top20_replication_fdr.csv")



