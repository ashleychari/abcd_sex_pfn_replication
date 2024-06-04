import os
import pandas as pd
import numpy as np

def get_accuracies(results_folder):
    time_dirs = os.listdir(results_folder)
    discovery_accuracies = []
    replication_accuracies = []
    for dir in time_dirs:
        if os.path.isdir(f"{results_folder}/{dir}"):
            metrics_table_filepath = f"{results_folder}/{dir}/2foldcv_metrics_table.csv"
            metrics_table = pd.read_csv(metrics_table_filepath)
            discovery_accuracy = metrics_table['accuracy'].values[0]
            replication_accuracy = metrics_table['accuracy'].values[1]
            discovery_accuracies.append(discovery_accuracy)
            replication_accuracies.append(replication_accuracy)

    accuracy_table = pd.DataFrame()
    accuracy_table['discovery_avg_accuracy'] = [np.mean(discovery_accuracies)]
    accuracy_table['replication_avg_accuracy'] = [np.mean(replication_accuracies)]

    accuracy_table.to_csv(f"{results_folder}/avg_accuracy_table.csv")


if __name__ == "__main__":
    results_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final_2"
    get_accuracies(results_folder)