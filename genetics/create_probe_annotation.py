import scipy.io
import pandas as pd

if __name__ == "__main__":
    roi_gene_mat = scipy.io.loadmat("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/AHBAprocessed/ROIxGene_Schaefer1000_INT.mat")
    gene_dictionary = {"gene": []}
    for array in roi_gene_mat['probeInformation']['GeneSymbol']:
        for gene_arrays in array:
            for gene_array in gene_arrays:
                gene_dictionary['gene'].append(gene_array[0][0])

    gene_df = pd.DataFrame.from_dict(gene_dictionary)
    gene_df.to_csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/genetics/GeneSymbol.csv", index=False)
    print("Job Done!")