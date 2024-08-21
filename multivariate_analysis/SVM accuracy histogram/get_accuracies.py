import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def get_metrics(data_for_ridge, results_folder, set, save_filename):
    # Use the data for ridge discovery subject ids as the ordering for the ids
    data_for_ridge_subject_ids = data_for_ridge['subjectkey'].values
    # also use data for ridge for sex (dont forget to transform the M/F)
    data_for_ridge_sex = data_for_ridge['sex'].values
    data_for_ridge_sex = [-1 if i == "M" else 1 for i in data_for_ridge_sex]

    accuracies_all_runs = []

    for i in range(1, 1001):
        time_folder_path = f"{results_folder}/time_{i}"
        predicted_values_array = np.load(f"{time_folder_path}/{set}_predicted_values_2_folds.npy", allow_pickle=True)
        predicted_values_fold1 = list(predicted_values_array[0])
        predicted_values_fold2 = list(predicted_values_array[1])
        predicted_values_stacked = predicted_values_fold1 + predicted_values_fold2

        # Load in the csv of the subject ids for each fold
        subject_ids_folds = pd.read_csv(f"{time_folder_path}/{set}_subject_ids_2_folds.csv")
        subject_ids_fold1 = list(subject_ids_folds['fold1_ids'].values)
        subject_ids_fold2 = list(subject_ids_folds['fold2_ids'].values)
        # Combine (stack) fold 1 and fold 2 decision values and id's
        subject_ids_stacked = subject_ids_fold1 + subject_ids_fold2
        if set == "replication":
            subject_ids_stacked = [id for id in subject_ids_stacked if not pd.isnull(id)]


        # Make df from stacked values (subject_id, decision_values)
        time_decision_values_df = pd.DataFrame()
        time_decision_values_df['subject_id'] = subject_ids_stacked
        time_decision_values_df['predicted_values'] = list(predicted_values_stacked)

        # Loop through data for ridge ids and get decision values in order
        #print(time_decision_values_df)
        print("Reordering predicted values....")
        predicted_values_in_order = []
        for subject_id in data_for_ridge_subject_ids:
            #print(subject_id)
            predicted_value = time_decision_values_df[time_decision_values_df['subject_id'] == subject_id]['predicted_values'].values[0]
            predicted_values_in_order.append(predicted_value)

        # Calculate accuracy for run
        accuracy = accuracy_score(data_for_ridge_sex, predicted_values_in_order)
        accuracies_all_runs.append(accuracy)

    accuracy_df = pd.DataFrame()
    accuracy_df['Accuracy'] = accuracies_all_runs
    accuracy_df.to_csv(save_filename)
    print(f"Average accuracy: {np.mean(accuracies_all_runs)}")


if __name__ == "__main__":
    discovery_data_for_ridge = pd.read_csv("/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_siblings_removed_071524.csv")
    res_multitimes_path = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/permutation_1000_times_081524"
    disc_avg_table_filename = "/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_permutation_accuracy_081524.csv"
    get_metrics(discovery_data_for_ridge, res_multitimes_path, "discovery", disc_avg_table_filename)

    replication_data_for_ridge = pd.read_csv("/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_siblings_removed_071524.csv")
    res_multitimes_path = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/permutation_1000_times_081524"
    rep_avg_table_filename = "/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_permutation_accuracy_081524.csv"
    get_metrics(replication_data_for_ridge, res_multitimes_path, "replication", rep_avg_table_filename)







        
    
