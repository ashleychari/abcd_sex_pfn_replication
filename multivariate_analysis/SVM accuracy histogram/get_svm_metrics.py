import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.stats as st

def get_metrics(data_for_ridge, results_folder, set):
    # Use the data for ridge discovery subject ids as the ordering for the ids
    data_for_ridge_subject_ids = data_for_ridge['subjectkey'].values
    # also use data for ridge for sex (dont forget to transform the M/F)
    data_for_ridge_sex = data_for_ridge['sex'].values
    data_for_ridge_sex = [-1 if i == "M" else 1 for i in data_for_ridge_sex]

    accuracies_all_runs = []
    sensitivity_all_runs = []
    specificity_all_runs = []

    for i in range(1, 101):
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
        #print("Reordering predicted values....")
        predicted_values_in_order = []
        for subject_id in data_for_ridge_subject_ids:
            #print(subject_id)
            predicted_value = time_decision_values_df[time_decision_values_df['subject_id'] == subject_id]['predicted_values'].values[0]
            predicted_values_in_order.append(predicted_value)

        # Calculate accuracy for run
        accuracy = accuracy_score(data_for_ridge_sex, predicted_values_in_order)
        accuracies_all_runs.append(accuracy)

        # Calculate sensitivity for run
        tn, fp, fn, tp = confusion_matrix(data_for_ridge_sex, predicted_values_in_order).ravel()
        sensitivity = tp / (tp + fn)
        sensitivity_all_runs.append(sensitivity)

        # Calculate specificity for run
        specificity = tn / (tn + fp)
        specificity_all_runs.append(specificity)

    return accuracies_all_runs, sensitivity_all_runs, specificity_all_runs
    
def make_avg_table(save_file, set_order, accuracies, sensitivities, specificities, ci):
    avg_table = pd.DataFrame()
    avg_table['set'] = set_order
    avg_table['Avg Accuracy'] = accuracies
    avg_table['Avg Sensitivity'] = sensitivities
    avg_table['Avg Specificity'] = specificities
    avg_table['95% CI'] = ci
    avg_table.to_csv(save_file)


    
if __name__ == "__main__":
    discovery_data_for_ridge = pd.read_csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/discovery and replication sample setup scripts/data/discovery_sample_siblings_removed_071524.csv")
    res_multitimes_path = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/res_100_times_roc_092624"
    avg_table_filename = "/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/multivariate_analysis/svm_092624_run/svm_100_times_092624_metrics.csv"
    accuracies_disc, sensitivity_disc, specificity_disc = get_metrics(discovery_data_for_ridge, res_multitimes_path, "discovery")
    #print(accuracies_disc)
    print(f"Discovery - avg acc: {np.mean(accuracies_disc)}, CI: {st.t.interval(confidence=0.95, df=len(accuracies_disc)-1, loc=np.mean(accuracies_disc), scale=st.sem(accuracies_disc))}")


    replication_data_for_ridge = pd.read_csv("/Users/ashfrana/Desktop/code/abcd_sex_pfn_replication/discovery and replication sample setup scripts/data/replication_sample_siblings_removed_071524.csv")
    accuracies_rep, sensitivity_rep, specificity_rep = get_metrics(replication_data_for_ridge, res_multitimes_path, "replication")
    print(f"Replication - avg acc: {np.mean(accuracies_rep)}, CI: {st.t.interval(confidence=0.95, df=len(accuracies_rep)-1, loc=np.mean(accuracies_rep), scale=st.sem(accuracies_rep))}")
    

    accuracies_100 = [np.mean(accuracies_disc), np.mean(accuracies_rep)]
    sensitivity_100 = [np.mean(sensitivity_disc), np.mean(sensitivity_rep)]
    specificity_100 = [np.mean(specificity_disc), np.mean(sensitivity_rep)]
    ci_100 = [str(st.t.interval(confidence=0.95, df=len(accuracies_disc)-1, loc=np.mean(accuracies_disc), scale=st.sem(accuracies_disc))), str(st.t.interval(confidence=0.95, df=len(accuracies_rep)-1, loc=np.mean(accuracies_rep), scale=st.sem(accuracies_rep)))]


    make_avg_table(avg_table_filename, ['discovery', 'replication'], accuracies_100, sensitivity_100, specificity_100, ci_100)

