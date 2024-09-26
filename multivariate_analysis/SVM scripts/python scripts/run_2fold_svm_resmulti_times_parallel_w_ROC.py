from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import os
from sklearn.metrics import confusion_matrix
import random

'''
Function to load in nonzero matrix
'''
def load_nonzero_mat(matrix_filename):
    features_nonzero_matrix = np.load(matrix_filename)
    return features_nonzero_matrix


'''
Function to add the covariates motion, age, and abcd_site
to the nonzero matrix
'''
# change parameters here
def add_covariates(data_for_ridge):
    print("Creating covariates matrix...")    
    all_covariates = pd.DataFrame()
    all_covariates['subject_id'] = data_for_ridge['subjectkey']
    all_covariates['meanFD'] = data_for_ridge['meanFD']
    all_covariates['age ']= data_for_ridge['interview_age'].values/12

    abcd_sites = data_for_ridge['abcd_site']
    # One hot encode abcd_site and add to covariates matrix
    site_dummies = pd.get_dummies(abcd_sites)
    covariates = pd.concat([all_covariates, site_dummies], axis=1)

    return covariates

def c_param_search(X_train, y_train, covariates, covariate_indices, random_state_seed, C_range):
    cs = []
    average_accuracies = []
    covariates_train = covariates[covariate_indices]

    for c_val in C_range:
        cs.append(c_val)
        fold_accuracies = []
        k_folds = KFold(n_splits=2, shuffle=True, random_state=random_state_seed) 
        for train_indices, val_indices in k_folds.split(X_train):
            X_train_cv = X_train[train_indices]
            X_val_cv = X_train[val_indices]
            y_train_cv = y_train[train_indices]
            y_val_cv = y_train[val_indices]

            nuisance_model = LinearRegression()
            nuisance_train = covariates_train[train_indices]
            nuisance_test = covariates_train[val_indices]
            nuisance_model.fit(nuisance_train, X_train_cv)

            X_train_nuisance_removed = X_train_cv - nuisance_model.predict(nuisance_train)
            X_val_nuisance_removed = X_val_cv - nuisance_model.predict(nuisance_test)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_nuisance_removed)
            X_val_scaled = scaler.transform(X_val_nuisance_removed)

            print("Fitting svm....")
            clf = svm.SVC(kernel='linear', C=c_val, random_state=42)
            clf.fit(X_train_scaled, y_train_cv)
            y_pred = clf.predict(X_val_scaled)
            fold_accuracies.append(accuracy_score(y_val_cv, y_pred))

    average_accuracies.append(np.mean(fold_accuracies))

    # Identify optimal c based on auc
    best_c_ind = np.argmax(average_accuracies)
    best_c = cs[best_c_ind]
    return best_c

def train_svm(X_train, y_train, X_test, y_test, covariates, indices_train, indices_test, best_c, subject_ids, results_dict):
    nuisance_model = LinearRegression()
    nuisance_train = covariates[indices_train]
    nuisance_test = covariates[indices_test]
    nuisance_model.fit(nuisance_train, X_train)
    X_train_nuisance_removed = X_train - nuisance_model.predict(nuisance_train)
    X_test_nuisance_removed = X_test - nuisance_model.predict(nuisance_test)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_nuisance_removed)
    X_test_scaled = scaler.transform(X_test_nuisance_removed)

    print("Fitting svm....")
    clf = svm.SVC(kernel='linear', C=best_c, random_state=42, probability=True)
    clf.fit(X_train_scaled, y_train)
    results_dict['coefs'].append(list(clf.coef_[0]))
    y_pred = clf.predict(X_test_scaled)
    results_dict['predicted_values'].append(y_pred)
    # results_dict['accuracies'].append(accuracy_score(y_test, y_pred))
    # results_dict['aucs'].append(roc_auc_score(y_test, clf.decision_function(X_test_scaled)))
    results_dict['decision_values'].append(clf.decision_function(X_test_scaled))
    results_dict['subject_ids_in_order'].append(np.array(subject_ids)[indices_test])
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # specificity = tn / (tn+fp)
    # results_dict['specificities'].append(specificity)
    return results_dict

# change parameters here
# data for ridge will be either discovery or replication data
def run_2fold_svm(matrix_filename, data_for_ridge, C_range, time_idx):
    features_nonzero_matrix = load_nonzero_mat(matrix_filename)

    # change function call
    covariates = add_covariates(data_for_ridge)
    
    X = features_nonzero_matrix

    # regress covariates out of features
    subject_ids = covariates['subject_id'].values
    covariates = covariates.drop(['subject_id'], axis=1)

    # Create y variable
    # sex_map = {"M": -1, "F": 1}
    # covariates['y'] = data_for_ridge['sex'].map(sex_map)
    # y = covariates['y'].values
    sex = data_for_ridge['sex'].values
    y = np.array([-1 if i == "M" else 1 for i in sex])

    # Convert covariates to numpy array for easier indexing
    covariates = np.array(covariates)


    # Split data into train and test
    random_state_seed = time_idx
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.5, random_state=random_state_seed)

    # Repeat 2 fold cross validation with train and test as a fold each
    results_dict = {'coefs': [], 'decision_values': [],
                    'subject_ids_in_order': [], 'predicted_values':[]}
    
    # Train as trainset then test as testset for fold 1
    best_c_fold1 = c_param_search(X_train, y_train, covariates, indices_train, random_state_seed, C_range)
    results_dict = train_svm(X_train, y_train, X_test, y_test, covariates, indices_train, indices_test, best_c_fold1, subject_ids, results_dict)


    # Test as trainset then train as testset as fold 2
    best_c_fold2 = c_param_search(X_test, y_test, covariates, indices_test, random_state_seed, C_range)
    results_dict = train_svm(X_test, y_test, X_train, y_train, covariates, indices_test, indices_train, best_c_fold2, subject_ids, results_dict)

    normalized_coefs = []
    for coef in results_dict['coefs']:
        coef = coef / np.linalg.norm(coef)
        normalized_coefs.append(coef)

    return results_dict['decision_values'], results_dict['predicted_values'], results_dict['subject_ids_in_order'], normalized_coefs, random_state_seed


def svm_wrapper(c_range_start, c_range_end, idx_time, results_folder):
    if os.path.isdir(f"{results_folder}/time_{idx_time}") == False:
        os.mkdir(f"{results_folder}/time_{idx_time}")

    c_range = range(c_range_start, c_range_end)
    c_values = [2**c for c in c_range]
    decision_values_discovery, predicted_values_discovery, subject_ids_discovery, coefs_discovery, random_state_seed_discovery = run_2fold_svm(discovery_nonzero_mat_filename, discovery_data_for_ridge, c_values, idx_time)
    decision_values_replication, predicted_values_replication, subject_ids_replication, coefs_replication, random_state_seed_replication = run_2fold_svm(replication_nonzero_mat_filename, replication_data_for_ridge, c_values, idx_time)
    # accuracy_discovery, auc_discovery, specificity_discovery, decision_values_discovery, subject_ids_discovery, coefs_discovery, random_state_seed_discovery = run_2fold_svm(discovery_nonzero_mat_filename, discovery_data_for_ridge, c_values)
    # accuracy_replication, auc_replication, specificity_replication, decision_values_replication, subject_ids_replication, coefs_replication, random_state_seed_replication = run_2fold_svm(replication_nonzero_mat_filename, replication_data_for_ridge, c_values)
    metrics_table = pd.DataFrame()
    metrics_table['set'] = ['discovery', 'replication']
    #metrics_table['accuracy'] = [accuracy_discovery, accuracy_replication]
    #metrics_table['auc'] = [auc_discovery, auc_replication]
    #metrics_table['specificity'] = [specificity_discovery, specificity_replication]
    metrics_table['random_state'] = [random_state_seed_discovery, random_state_seed_replication]
    metrics_table.to_csv(f"{results_folder}/time_{idx_time}/2foldcv_metrics_table.csv")
    # Save decision values for discovery set
    np.save(f"{results_folder}/time_{idx_time}/discovery_decision_values_2_folds", np.array(decision_values_discovery))

    # Save decision values for replication set
    np.save(f"{results_folder}/time_{idx_time}/replication_decision_values_2_folds", np.array(decision_values_replication))

    # Save predicted values for discovery set
    np.save(f"{results_folder}/time_{idx_time}/discovery_predicted_values_2_folds", np.array(predicted_values_discovery))

    # Save predicted values for replication set
    np.save(f"{results_folder}/time_{idx_time}/replication_predicted_values_2_folds", np.array(predicted_values_replication))



    # Save subject ids for both folds for discovery
    # make sure lists are the same len
    ### Move into a function #####
    subject_ids_discovery_fold1 = list(subject_ids_discovery[0])
    subject_ids_discovery_fold2 = list(subject_ids_discovery[1])
    if len(subject_ids_discovery_fold1) != len(subject_ids_discovery_fold2):
        # check which list is less than and by how much
        list1_lt = len(subject_ids_discovery_fold1) < len(subject_ids_discovery_fold2)
        list2_lt = len(subject_ids_discovery_fold2) < len(subject_ids_discovery_fold1)

        if list1_lt:
            diff = len(subject_ids_discovery_fold2) - len(subject_ids_discovery_fold1)
            for i in range(diff):
                subject_ids_discovery_fold1.append("NA")

        if list2_lt:
            diff = len(subject_ids_discovery_fold1) - len(subject_ids_discovery_fold2)
            for i in range(diff):
                subject_ids_discovery_fold2.append("NA")


    subject_id_table_discovery = pd.DataFrame()
    subject_id_table_discovery['fold1_ids'] = subject_ids_discovery_fold1
    subject_id_table_discovery['fold2_ids'] = subject_ids_discovery_fold2
    subject_id_table_discovery.to_csv(f"{results_folder}/time_{idx_time}/discovery_subject_ids_2_folds.csv")


    # Save subject ids for both folds for replication
    # make sure lists are the same len
    ### move into function ####
    subject_ids_replication_fold1 = list(subject_ids_replication[0])
    subject_ids_replication_fold2 = list(subject_ids_replication[1])
    if len(subject_ids_replication_fold1) != len(subject_ids_replication_fold2):
        # check which list is less than and by how much
        list1_lt = len(subject_ids_replication_fold1) < len(subject_ids_replication_fold2)
        list2_lt = len(subject_ids_replication_fold2) < len(subject_ids_replication_fold1)

        if list1_lt:
            diff = len(subject_ids_replication_fold2) - len(subject_ids_replication_fold1)
            for i in range(diff):
                subject_ids_replication_fold1.append("NA")

        if list2_lt:
            diff = len(subject_ids_replication_fold1) - len(subject_ids_replication_fold2)
            for i in range(diff):
                subject_ids_replication_fold2.append("NA")


    subject_id_table_replication = pd.DataFrame()
    subject_id_table_replication['fold1_ids'] = subject_ids_replication_fold1
    subject_id_table_replication['fold2_ids'] = subject_ids_replication_fold2
    subject_id_table_replication.to_csv(f"{results_folder}/time_{idx_time}/replication_subject_ids_2_folds.csv")



    coefs_discovery_table = pd.DataFrame()
    i = 1
    for coef_fold in coefs_discovery:
        coefs_discovery_table[f"fold_{i}_coefs"] = coef_fold
        i += 1
    coefs_discovery_table.to_csv(f"{results_folder}/time_{idx_time}/2foldcv_discovery_coefs.csv")

    coefs_replication_table = pd.DataFrame()
    i = 1
    for coef_fold in coefs_replication:
        coefs_replication_table[f"fold_{i}_coefs"] = coef_fold
        i += 1
        
    coefs_replication_table.to_csv(f"{results_folder}/time_{idx_time}/2foldcv_replication_coefs.csv")
    
if __name__ == "__main__":
    discovery_nonzero_mat_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery_siblings_removed.npy'
    replication_nonzero_mat_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_siblings_removed.npy'
    discovery_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_siblings_removed_071524.csv')
    replication_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_siblings_removed_071524.csv')
    discovery_nonzero_indices = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery_siblings_removed_nonzero_indices.csv')
    replication_nonzero_indices = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_siblings_removed_nonzero_indices.csv')
    results_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_roc_092624"
    c_range_start = int(sys.argv[1])
    c_range_end = int(sys.argv[2])
    time_idx = int(sys.argv[3])
    svm_wrapper(c_range_start, c_range_end, time_idx, results_folder)