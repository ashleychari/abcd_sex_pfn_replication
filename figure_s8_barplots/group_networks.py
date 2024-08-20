import pandas as pd
import sys

if __name__ == "__main__":
    network_data_filename = sys.argv[1]
    save_filename = sys.argv[2]
    test_type = sys.argv[3]

    network_df = pd.read_csv(network_data_filename)

    # subset by gender
    male_subset = network_df[network_df['sex'] == 'male']
    female_subset = network_df[network_df['sex'] == 'female']

    if test_type == "svm":
        male_subset_grouped = male_subset[['netColor', 'weights']].groupby(by='netColor').sum()
        male_subset_grouped['sex'] = 'male'

        female_subset_grouped = female_subset[['netColor', 'weights']].groupby(by='netColor').sum()
        female_subset_grouped['sex'] = 'female'

    else:
        male_subset_grouped = male_subset[['netColor', 'vertecies']].groupby(by='netColor').sum()
        male_subset_grouped['sex'] = 'male'

        female_subset_grouped = female_subset[['netColor', 'vertecies']].groupby(by='netColor').sum()
        female_subset_grouped['sex'] = 'female'
    

    network_grouped_df = pd.concat([female_subset_grouped, male_subset_grouped], axis=0)

    network_grouped_df.to_csv(save_filename)


