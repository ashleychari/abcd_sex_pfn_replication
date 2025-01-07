import os
import pandas as pd
import numpy as np


def identify_sig_networks(folder_path):
    sig_networks = []
    for network in range(1, 18):
        network_path = f"{folder_path}/network_{network}.csv"
        network_df = pd.read_csv(network_path)
        if not all(x == 0 for x in network_df['0'].values):
            sig_networks.append(network)

    return sig_networks

def write_sig_networks(network_paths):
    for folder in os.listdir(network_paths):
        summary_file = f"{network_paths}/{folder}/sig_networks.txt"
        f = open(summary_file, "w+")
        folder_path = f"{network_paths}/{folder}"
        sig_networks = identify_sig_networks(folder_path)
        f.write(f"Significant networks: {sig_networks}")
        f.close()


if __name__ == "__main__":
    disc_network_paths = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/individual_network_maps/discovery"
    rep_network_paths = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/univariate_analysis/pubertal_analyses/individual_network_maps/replication"

    write_sig_networks(disc_network_paths)
    write_sig_networks(rep_network_paths)
    

    
        
        
            