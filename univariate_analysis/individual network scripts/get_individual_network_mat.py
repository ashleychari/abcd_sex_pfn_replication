import pandas as pd
import os

def get_individual_network_mat(all_networks_mat, network_num):
    individual_network_mat = all_networks_mat.iloc[network_num-1, :]
    filename = f"/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/individual network scripts/individual_network_matrices/individual_network_{network_num}_mat.csv"
    individual_network_mat.to_csv(filename, header=['0'], index=False)
    print(f"Network {network_num} matrix created!")

if __name__ == "__main__":
    discovery_gams_all_networks_mat = pd.read_csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/univariate_analysis_results/discovery_barplot_all_networks_mat.csv")
    if "Unnamed: 0" in discovery_gams_all_networks_mat.columns:
        discovery_gams_all_networks_mat = discovery_gams_all_networks_mat.drop(['Unnamed: 0'], axis=1)


    for i in range(1, 18):
        get_individual_network_mat(discovery_gams_all_networks_mat, i)
