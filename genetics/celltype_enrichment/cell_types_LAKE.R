library(R.matlab)
library(dplyr)
library(magrittr)
library(tibble)
library(ggplot2)
library(data.table)
library(ggplot2)
library(hexbin)
library(plotrix)
library(matrixStats)

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

# brain imaging data
data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_uncorrected_discovery_LH.fsaverage5.func.gii')
#data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_uncorrected_replication_LH.fsaverage5.func.gii')
#data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_fdr_discovery_LH.fsaverage5.func.gii')
#data_brain1 <- readgii('/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/gams_gii_files/gams_fdr_replication_LH.fsaverage5.func.gii')
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
roislh <- rois[1:500,]

img_geneMatChrom_roi_500 <- merge(roislh, img_geneMatChrom_500, by.x = "RoiLabel", by.y= "parcNum",all = FALSE)
img_geneMatChrom_roi_500_comp <- subset(img_geneMatChrom_roi_500, !is.nan(img_geneMatChrom_roi_500[[10]])) ##if add more columns, change "10". This is the complete DF. Any subsets below should come from here/match this order.


data_parc_lh_sort_500_comp <- subset(img_geneMatChrom_roi_500_comp, select=wt_mean)


othercols <- c(colnames(rois), "lh_schaefer1000", "wt_mean")
geneMatChrom_500_comp <- img_geneMatChrom_roi_500_comp[(length(othercols)+1):(length(colnames(img_geneMatChrom_roi_500_comp)))]







###read in gene data
#check cell enrichement 
cell3 <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics_files/cellTypes/Lake18_celltypes.csv")
genebrain <- read.csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics_files/cellTypes/brain_genes_HPA.csv", header = FALSE)
cell2 <- merge(cell3, genebrain, by.x="Gene", by.y="V1")

cellRefined <- cell2$Cluster[cell2$Gene %in% gene$V1] #which values of cell2$Gene are in gene$V1 
geneRefined <- cell2$Gene[cell2$Gene %in% gene$V1] 
geneMatCell_500 <-geneMatChrom_500_comp[,match(geneRefined,colnames(geneMatChrom_500_comp),nomatch = 0)] #target gene first. where in #2 is #1


geneRefined_df <- as.data.frame(geneRefined)
cellRefined_df <- as.data.frame(cellRefined)
cellGeneRefined <- cbind(cellRefined_df, geneRefined_df)





t2 <- sapply(1:length(geneMatCell_500), function(x) cor(data_parc_lh_sort_500_comp,geneMatCell_500[,x]))
colt2<- as.vector(cellGeneRefined$geneRefined)
dft2 <- as.data.frame(cbind(colt2, t2))
colnames(dft2)<- c("gene", "corr")

df3a <- merge(dft2, cellGeneRefined, by.x="gene", by.y="geneRefined")
df3 <- unique(df3a) 



#plot for classical method, using df3
#Rank these 4558 observations based on correlation
df3$corrSigRanked <- rank(df3$corr)
#Calculate median ranking for each cell type
df3$cellRefined <- as.factor(df3$cellRefined)
t3 <- sapply(levels(df3$cellRefined), function(x) median(df3$corrSigRanked[which(df3$cellRefined==x)],na.rm = TRUE)) # which(chromRefined==x) --  get an index of all the trues,,,   indexing out of t2Ranked where chromRefined==x.... find the median of t2Ranked for each of these levels
#subtract the overall median rank of the 4965 observations from the median rank for each celltype 
t4<- t3 - median(df3$corrSigRanked) # for the average correlation ranking for each cellType, how far away it is from the median
t4
t5a <- as.data.frame(t4)
setDT(t5a, keep.rownames = "cellType")
t5a$cellType <- as.factor(t5a$cellType)
t5 <- t5a




t3se <- sapply(levels(df3$cellRefined), function(x) std.error(df3$corrSigRanked[which(df3$cellRefined==x)],na.rm = TRUE)) # which(chromRefined==x) --  get an index of all the trues,,,   indexing out of t2Ranked where chromRefined==x.... find the median of t2Ranked for each of these levels
t3se
t5se <- as.data.frame(t3se)
setDT(t5se, keep.rownames = "cellType")
t5se$cellType <- as.factor(t5$cellType)

perm.pval <- data.frame(matrix(NA,nrow=length(t5$cellType),ncol=3))
colnames(perm.pval)<- c("cellType", "NumGenes", "pval")

t5vec <- as.list(as.character(t5$cellType))
df3gene_unique<- unique(df3$gene)

for (j in 1:length(t5$cellType)){
  cellType_tmp <- t5vec[j]
  All_tmp <- subset(df3, cellRefined == cellType_tmp)
  genenum <- length(All_tmp$gene)
  
  if (genenum > 0) {
    null.dist.medians <-data.frame(matrix(NA,nrow=1000,ncol=1))
    colnames(null.dist.medians)<- "median_rank"
    for (i in 1:1000){
      set.seed(i)
      genelisttmp <- sample(df3gene_unique, genenum, replace = FALSE, prob = NULL)
      dfgenestmp <- filter(df3, gene %in% genelisttmp)
      median_rank <- median(dfgenestmp$corrSigRanked)
      null.dist.medians[i, 1] <- median_rank
    }
  } 
  newdata <- subset(t5, cellType == cellType_tmp)
  print(newdata)
  if (newdata$t4 > 0){
    pval<- (sum(null.dist.medians$median_rank > median(All_tmp$corrSigRanked)))/1000
  }
  if (newdata$t4 < 0){
    pval<- (sum(null.dist.medians$median_rank < median(All_tmp$corrSigRanked)))/1000
  }
  perm.pval[j, 1]  <- cellType_tmp
  perm.pval[j, 2] <- genenum
  perm.pval[j, 3] <- pval
}


df_enrich_pval <- merge(t5, perm.pval, by = "cellType")
df_enrich_pval1 <- df_enrich_pval[with(df_enrich_pval, order(-t4)),]



t8 <- merge (t5, t5se, by = "cellType")

colnames(t8)<- c("cellType", "Rank", "se")



BorderColor <- c("#FF5F61", "#FF5F61", "#DBA73D", "#00CACD", "#00CACD", 
                 "#00CACD", "#00CACD", "#00CACD", "#00CACD", "#00CACD",
                 "#00CACD", "#00CACD", "#00CACD", "#00CACD", "#00CACD", 
                 "#00CACD", "#2BEA20", "#25B8FF", "#25B8FF", "#25B8FF",
                 "#25B8FF", "#25B8FF", "#25B8FF", "#25B8FF", "#25B8FF", 
                 "#25B8FF", "#25B8FF", "#25B8FF", "#8CBF38", "#D98AFF", 
                 "#FF70D7", "#FF70D7", "#E619C2", "#FF70D7", "#FF70D7");






#fillcolor <- c("#FF5F61", "#FFFFFF", "#FFFFFF", "#00CACD", "#00CACD", "#FFFFFF", "#00CACD", "#FFFFFF", "#00CACD", "#00CACD", 
#               "#FFFFFF", "#FFFFFF", "#00CACD", "#FFFFFF", "#00CACD", 
#               "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", 
#               "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", 
#               "#FFFFFF", "#FFFFFF", "#FFFFFF", "#8CBF38", "#D98AFF", 
#               "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FFFFFF", "#FF70D7");

# what we want: if the lines are dashed then the fillcolor should be white, otherwise it should be 
# the same color as the border color
# how to do this: Can go through lineType df and match that celltype's color from the 
# border color to the fillcolor if its solid, otherwise fillcolor will be white

t9 <- cbind(t8, BorderColor)

df_enrich_pval1$LineType <- with(df_enrich_pval1, ifelse(pval > 0.05, 'dashed', 'solid'))
LineType_df <- df_enrich_pval1[,c("cellType", "LineType")]

fillColor <- c()
for (i in (1:length(LineType_df$cellType))){
  celltype <- LineType_df$cellType[i]
  linetype <- LineType_df$LineType[i]
  if (linetype == 'dashed'){
    # fillcolor is white 
    fillColor <- append(fillColor, "#FFFFFF")
  }
  else{
    correctFill <- subset(t9$BorderColor, t9$cellType == celltype)
    print(correctFill)
    fillColor <- append(fillColor, correctFill)
  }
}

LineType_df$fillColor <- fillColor
#LineType <- c("solid", "dashed", "dashed", "solid", "solid", 
#              "dashed", "solid", "dashed", "solid", "solid", 
#              "dashed", "dashed", "solid", "dashed", "solid", 
#              "dashed", "dashed", "dashed", "dashed", "dashed", 
#              "dashed", "dashed", "dashed", "dashed", "dashed", 
#              "dashed", "dashed", "dashed", "solid", "solid", 
#              "dashed", "dashed", "dashed", "dashed", "solid");


t9 <- merge(t9, LineType_df, by='cellType')
#make color label an ordered factor so ggplot can match the color to network
t9$fillcolor <- factor(t9$fillColor, ordered =T)
t9$BorderColor <- factor(t9$BorderColor, ordered =T)
t9$LineType <- factor(t9$LineType, ordered =T)


t9 <- t9 %>% 
  mutate(cellType = as.factor(cellType),fillcolor=as.character(fillcolor))%>%
  mutate(cellType = as.factor(cellType),BorderColor=as.character(BorderColor))%>%
  mutate(cellType = as.factor(cellType),LineType=as.character(LineType))%>%
  mutate(cellType = reorder(cellType,Rank))%>%
  arrange (cellType)
colormap=t9 %>% select(cellType,fillcolor)%>%unique()
linemap=t9 %>% select(cellType,LineType)%>%unique()
bordermap=t9 %>% select(cellType,BorderColor)%>%unique()


overall_se <- std.error(df3$corrSigRanked)


#figure S2
#tiff("/Users/sheilash/Desktop/projects/pfn_sex_diff/paper/figures/genetics/cellType_enrichments.tiff", width = 5, height = 5, units = 'in', res = 300)

p<-ggplot(data=t9, aes(x=cellType, y=Rank)) +
  geom_bar(stat="identity", fill=colormap$fillcolor ,
           colour = bordermap$BorderColor, linetype = linemap$LineType, width = 0.8) + geom_pointrange(aes(ymin=Rank-se, ymax=Rank+se), size=0, linetype = linemap$LineType, position=position_dodge(.9)) +
  geom_point(data=t9, aes(y=Rank, x=cellType), size = 0.8) +
  coord_flip() + xlab("Cell Type") + ylab("Enrichment") +
  geom_hline(aes(yintercept = as.numeric(0))) +
  #ylim(-2799, 1010) +
  #scale_y_discrete(breaks = c(-2000, -1000, 1000)) +
  theme(legend.position="none") + theme(axis.text.x = element_text(size= 10), axis.text.y = element_text(size= 10, color = "grey30"), axis.title=element_text(size = 18))

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/genetics/cell_type_enrichments/high_res/gams_uncorrected_discovery.png", height=5, width=5, dpi=300)



write.csv(df_enrich_pval1, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/celltype_enrichment/gams_uncorrected_discovery_celltypes_pvalues.csv")


df3_gene_astro <- subset(df3, df3$cellRefined=='Ast')
df3_gene_astro_ranked <- df3_gene_astro[order(df3_gene_astro$corr, decreasing=TRUE), ]
df3_gene_astro_top20 <- df3_gene_astro_ranked[1:20,]
write.csv(df3_gene_astro_top20, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/top20_enrichment_tables/astro_top20_uncorrected_discovery.csv")
#df3_gene_ranked_X <- df3_gene_X[order(df3_gene_X$corr, decreasing=TRUE), ]
#df3_gene_X_ranked_top20 <- df3_gene_ranked_X[1:20, ]


df3_gene_ex5b <- subset(df3, df3$cellRefined=='Ex5b')
df3_gene_ex5b_ranked <- df3_gene_ex5b[order(df3_gene_ex5b$corr, decreasing=TRUE), ]
df3_gene_ex5b_top20 <- df3_gene_ex5b_ranked[1:20,]
write.csv(df3_gene_ex5b_top20, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/top20_enrichment_tables/ex5b_top20_uncorrected_discovery.csv")


df3_gene_ex1 <- subset(df3, df3$cellRefined=='Ex1')
df3_gene_ex1_ranked <- df3_gene_ex1[order(df3_gene_ex1$corr, decreasing=TRUE), ]
df3_gene_ex1_top20 <- df3_gene_ex1_ranked[1:20,]
write.csv(df3_gene_ex1_top20, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/top20_enrichment_tables/ex1_top20_uncorrected_discovery.csv")


df3_gene_oli <- subset(df3, df3$cellRefined=='Oli')
df3_gene_oli_ranked <- df3_gene_oli[order(df3_gene_oli$corr, decreasing=TRUE), ]
df3_gene_oli_top20 <- df3_gene_oli_ranked[1:20,]
write.csv(df3_gene_oli_top20, "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/genetics/top20_enrichment_tables/oli_top20_uncorrected_discovery.csv")



