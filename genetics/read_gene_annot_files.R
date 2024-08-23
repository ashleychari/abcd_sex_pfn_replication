library(freesurferformats)
library(freesurfer)
annot_file <- read.fs.annot("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/data/genes/parcellations/lh.Schaefer1000_7net.annot")

lh_Schaefer1000_7net_cttable <- annot_file$colortable$table[,5]
write.table(lh_Schaefer1000_7net_cttable, "/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/lh_Schaefer1000_7net_cttable.csv", row.names = FALSE, col.names = FALSE)

annot_file_2 <- read_annotation("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/data/genes/parcellations/lh.Schaefer1000_7net.annot")
L <- annot_file_2$label
write.table(L, "/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/lh_Schaefer1000_7net_L.csv",  row.names = FALSE, col.names = FALSE)

