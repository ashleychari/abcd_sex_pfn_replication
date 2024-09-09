
library(R.matlab)
library(dplyr)
library(magrittr)
library(tibble)
library(ggplot2)
library(data.table)
library(ggpattern)



theme_set(theme_classic(base_size = 12))

sums_all_rank <- read.csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/figure_s8_barplots/grouped_network_tables/svm_replication_grouped_networks.csv")

network <- c(1, 2, 3, 4, 5, 6, 7)
netColor <- c("#AF33AD", "#4E31A8", "#7499C2", "#00A131", "#E443FF",
              "#F5BA2E", "#E76178")
netLabel <- c("VS", "AU", "SM", "DA", "VA", "FP", "DM")

netColorDf <- cbind(network, netColor, netLabel)


#combine df of sum of abs value of weights with color table
sums_all_col <- merge(sums_all_rank, netColorDf, by = "netColor")

#make color label an ordered factor so ggplot can match the color to network

#sums_all_col$netrank <- rank(sums_all_col$vertecies)
sums_all_col$netrank <- rank(sums_all_col$weights)
sums_all_col$netColorF <- factor(sums_all_col$netColor, ordered =T)
sums_all_col$netLabel <- factor(sums_all_col$netLabel, ordered=T)

sums_all_col_rank <- sums_all_col


#sums_all_col_rank <- sums_all_col_rank %>% 
#  mutate(network = as.factor(network),netColorF=as.character(netColorF))%>%
#  mutate(network = reorder(network,+vertecies))%>%
#  arrange (netrank)


sums_all_col_rank <- sums_all_col_rank %>% 
  mutate(network = as.factor(network),netColorF=as.character(netColorF))%>%
  mutate(network = reorder(network,+weights))%>%
  arrange (netrank)


colormap=sums_all_col_rank %>% select(network,netColorF)%>%unique()
labelmap=sums_all_col_rank %>% select(network,netLabel)%>%unique()
colormap <- colormap[order(colormap$network), ]
labelmap <- labelmap[order(labelmap$network), ]


#netname <- c("VSN", "AUN", "SMN", "DAN", "VA", "FP", "DM")

gam_plot <- ggplot(sums_all_col_rank, aes(x = reorder(network, +vertecies), y = vertecies, fill=network, alpha=factor(sex)))+
  geom_col(position = "stack", color="black", width=0.7) + xlab("Network") + ylab("Significant Vertices")+
  scale_alpha_manual(values = c("male"=0.5, "female"=1), guide='none') + scale_fill_manual(values = colormap$netColorF) + scale_y_continuous(breaks=seq(0, 30000, 10000), limits=c(0, 30000), expand = c(0, 0)) +
  geom_col_pattern(aes(pattern_alpha = sex),
                   fill = NA, pattern = 'stripe',
                   pattern_fill = "black",
                   pattern_angle = 45,
                   pattern_density = 0.05,
                   pattern_spacing = 0.03,
                   pattern_key_scale_factor = 0.5,
                   width=0.7) + scale_pattern_alpha_discrete(range = c(0,0.5), labels =c("Female", "Male")) +
  theme(legend.position="none") + theme(axis.text.x = element_text(size= 12), axis.text.y = element_text(size= 12), axis.title = element_text(size=18)) + scale_x_discrete(label=labelmap$netLabel)
gam_plot


ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/fig_s8/gams_uncorrected_discovery.png", gam_plot, width=4.8, height=3.5, dpi=300)

#tiff("/Users/sheilash/Desktop/projects/pfn_sex_diff/paper/figures/barplots/SVM_weights_barplot.tiff", width = 4.8, height = 3.5, units = 'in', res = 300)
svm_plot <- ggplot(sums_all_col_rank, aes(x = reorder(network, +weights), y = weights, fill=network, alpha=factor(sex)))+
  geom_col(position = "stack", color="black", width=0.7) + xlab("Network") + ylab("Weights")+
  scale_alpha_manual(values = c("male"=0.5, "female"=1), guide='none') + scale_fill_manual(values = colormap$netColorF) + scale_y_continuous(breaks=seq(0, 600, 100), limits=c(0, 600), expand = c(0, 0)) +
  geom_col_pattern(aes(pattern_alpha = sex),
                    fill = NA, pattern = 'stripe',
                    pattern_fill = "black",
                    pattern_angle = 45,
                    pattern_density = 0.05,
                    pattern_spacing = 0.03,
                    pattern_key_scale_factor = 0.5,
                   width=0.7) + scale_pattern_alpha_discrete(range = c(0,0.5), labels =c("Female", "Male")) +
  theme(legend.position="none") + theme(axis.text.x = element_text(size= 12), axis.text.y = element_text(size= 12), axis.title = element_text(size=18)) + scale_x_discrete(label=labelmap$netLabel)

svm_plot 

ggsave("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/finalized_figs/fig_s8/svm_replication.png", svm_plot, width=4.8, height=3.5, dpi=300)
