import pandas as pd
import numpy as np

def create_discovery_or_replication_set(data_for_ridge, group):
    # Subset data for either discovery (matched_group=1) or replication (matched_group=2)
    data = data_for_ridge[data_for_ridge['matched_group'] == group]

    # randomly select one subject from the group of people with the same rel_family_id to remove siblings
    siblings_dict = {}
    for fam_id in data['rel_family_id'].values:
        siblings_dict[fam_id] = []

    # Add all of the siblings to each family id list in dictionary
    for i in range(len(data['rel_family_id'])):
        siblings_dict[data['rel_family_id'].values[i]].append(data['subjectkey'].values[i])

    # Randomly select one sibling from each list
    np.random.seed(42)
    selected_ids = []
    for fam_id, sibling_list in siblings_dict.items():
        selected_sibling = np.random.choice(sibling_list)
        selected_ids.append(selected_sibling)

    # Only get the selected subject ids
    data_removed_siblings = data[data['subjectkey'].isin(selected_ids)]

    return data_removed_siblings


if __name__ == "__main__":
    # Read in data for ridge (created using Arielle's script)
    data_for_ridge_ariels_code = pd.read_csv("/Users/ashfrana/Desktop/code/ABCD GAMs replication/data_for_ridge_030824.csv")

    # Create discovery data siblings removed sample
    discovery_siblings_removed = create_discovery_or_replication_set(data_for_ridge_ariels_code, 1)

    # Create replication data siblings removed sample
    replication_siblings_removed = create_discovery_or_replication_set(data_for_ridge_ariels_code, 2)

    # Save datasets
    discovery_siblings_removed.to_csv("discovery_sample_siblings_removed_071524.csv")
    replication_siblings_removed.to_csv("replication_sample_siblings_removed_071524.csv")
    
    