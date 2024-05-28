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
def add_covariates(data_for_ridge_discovery, data_for_ridge_replication):
    print("Creating covariates matrix...")
    data_for_ridge = pd.concat([data_for_ridge_discovery, data_for_ridge_replication], axis=0)

    all_covariates = pd.DataFrame()
    all_covariates['subject_id'] = data_for_ridge['subjectkey']
    all_covariates['matched_group'] = data_for_ridge['matched_group']
    all_covariates['meanFD'] = data_for_ridge['meanFD']
    all_covariates['age ']= data_for_ridge['interview_age'].values/12

    abcd_sites = data_for_ridge['abcd_site']
    # One hot encode abcd_site and add to covariates matrix
    site_dummies = pd.get_dummies(abcd_sites)
    covariates = pd.concat([all_covariates, site_dummies], axis=1)
    covariates_discovery = covariates[covariates['matched_group'] == 1]
    covariates_replication = covariates[covariates['matched_group'] == 2]
    print(f"len(subject_ids) in covariates df: {len(covariates_discovery['subject_id'])}")
    print(covariates_discovery)

    print(f"len(subject_ids) in covariates df: {len(covariates_replication['subject_id'])}")
    print(covariates_replication)


    return covariates_discovery, covariates_replication

def c_param_search(C_range, covariates, dataset, indices):
    cs = []
    average_accuracies = []
    average_aucs = []
    covariates_train = covariates[indices]
    print(f"len X_train: {len(dataset)}")
    print(f"len covariates_train: {len(covariates_train)}")
    for i in C_range:
        cs.append(i)
        fold_accuracies = []
        fold_aucs = []
        k_folds = KFold(n_splits=2, shuffle=True)
        for train_indices, test_indices in k_folds.split(dataset):
            X_train_cv = dataset[train_indices]
            X_val_cv = dataset[test_indices]
            y_train_cv = dataset[train_indices]
            y_val_cv = dataset[test_indices]

            nuisance_model = LinearRegression()
            nuisance_train = covariates_train[train_indices]
            nuisance_test = covariates_train[test_indices]
            nuisance_model.fit(nuisance_train, X_train_cv)

            X_train_nuisance_removed = X_train_cv - nuisance_model.predict(nuisance_train)
            X_val_nuisance_removed = X_val_cv - nuisance_model.predict(nuisance_test)

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_nuisance_removed)
            X_val_scaled = scaler.transform(X_val_nuisance_removed)

            print("Fitting svm....")
            clf = svm.SVC(kernel='linear', C=i, random_state=42)
            clf.fit(X_train_scaled, y_train_cv)
            y_pred = clf.predict(X_val_scaled)
            fold_accuracies.append(accuracy_score(y_val_cv, y_pred))
            fold_aucs.append(roc_auc_score(y_val_cv, y_pred))

    average_accuracies.append(np.mean(fold_accuracies))
    average_aucs.append(np.mean(fold_aucs))

    # Identify optimal c based on auc
    best_c_ind = np.argmax(average_accuracies)
    best_c = cs[best_c_ind]
    return best_c



def run_2fold_svm(matrix_filename, discovery_data, replication_data, matched_group, C_range):
    features_nonzero_matrix = load_nonzero_mat(matrix_filename)
    data_for_ridge = pd.concat([discovery_data, replication_data], axis=0)
    covariates_discovery, covariates_replication = add_covariates(discovery_data, replication_data)
    if matched_group == "discovery":
        covariates = covariates_discovery
    else:
        covariates = covariates_replication
    X = features_nonzero_matrix

    # regress covariates out of features
    covariates = covariates.drop(['subject_id'], axis=1)
    covariates = np.array(covariates)

    if matched_group == "discovery":
        sex = data_for_ridge[data_for_ridge['matched_group'] == 1]['sex'].values
    else:
        sex = data_for_ridge[data_for_ridge['matched_group'] == 2]['sex'].values
    y = np.array([1 if i == "M" else -1 for i in sex]) # set Male=1, female=-1

    # Split data into train and test
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.5)

    # Repeat 2 fold cross validation with train and test as a fold each
    coefs = []
    accuracies = []
    aucs = []
    fprs = []
    tprs = []
    specificities = []
    
    # Train as trainset then test as testset for fold 1
    best_c = c_param_search(C_range, covariates, X_train, indices_train)

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
    clf = svm.SVC(kernel='linear', C=best_c, random_state=42)
    clf.fit(X_train_scaled, y_train)
    coefs.append(list(clf.coef_[0]))
    y_pred = clf.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, y_pred))
    aucs.append(roc_auc_score(y_test, y_pred))
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    fprs.append(fpr)
    tprs.append(tpr)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn+fp)
    specificities.append(specificity)


    # Test as trainset then train as testset as fold 2
    best_c = c_param_search(C_range, covariates, X_test, indices_test)

    nuisance_model = LinearRegression()
    nuisance_train = covariates[indices_train]
    nuisance_test = covariates[indices_test]
    nuisance_model.fit(nuisance_test, X_test)
    X_train_nuisance_removed = X_train - nuisance_model.predict(nuisance_train)
    X_test_nuisance_removed = X_test - nuisance_model.predict(nuisance_test)

    scaler = MinMaxScaler()
    X_test_scaled = scaler.fit_transform(X_test_nuisance_removed)
    X_train_scaled = scaler.transform(X_train_nuisance_removed)

    print("Fitting svm....")
    clf = svm.SVC(kernel='linear', C=best_c, random_state=42)
    clf.fit(X_test_scaled, y_test)
    coefs.append(list(clf.coef_[0]))
    y_pred = clf.predict(X_train_scaled)
    accuracies.append(accuracy_score(y_train, y_pred))
    aucs.append(roc_auc_score(y_train, y_pred))
    fpr, tpr, _ = metrics.roc_curve(y_train, y_pred)
    fprs.append(fpr)
    tprs.append(tpr)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    specificity = tn / (tn+fp)
    specificities.append(specificity)

    normalized_coefs = []
    for coef in coefs:
        coef = coef / np.linalg.norm(coef)
        normalized_coefs.append(coef)

    return np.mean(accuracies), np.mean(aucs), np.mean(fpr), np.mean(tpr), np.mean(specificities), normalized_coefs


def svm_wrapper(c_range_start, c_range_end, idx_time):
    if os.path.isdir(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final/time_{idx_time}") == False:
        os.mkdir(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final/time_{idx_time}")

    c_range = range(c_range_start, c_range_end)
    c_values = [2**i for i in c_range]
    accuracy_discovery, auc_discovery, fpr_discovery, tpr_discovery, specificity_discovery, coefs_discovery = run_2fold_svm(discovery_nonzero_mat_filename, discovery_data_for_ridge, replication_data_for_ridge, "discovery", c_values)
    accuracy_replication, auc_replication, fpr_replication, tpr_replication, specificity_replication, coefs_replication = run_2fold_svm(replication_nonzero_mat_filename, discovery_data_for_ridge, replication_data_for_ridge, "replication", c_values)
    metrics_table = pd.DataFrame()
    metrics_table['set'] = ['discovery', 'replication']
    metrics_table['accuracy'] = [accuracy_discovery, accuracy_replication]
    metrics_table['auc'] = [auc_discovery, auc_replication]
    metrics_table['tpr'] = [tpr_discovery, tpr_replication]
    metrics_table['fpr'] = [fpr_discovery, fpr_replication]
    metrics_table['specificity'] = [specificity_discovery, specificity_replication]
    metrics_table.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final/time_{idx_time}/2foldcv_metrics_table.csv")

    coefs_discovery_table = pd.DataFrame()
    i = 1
    for coef_fold in coefs_discovery:
        coefs_discovery_table[f"fold_{i}_coefs"] = coef_fold
        i += 1
    coefs_discovery_table.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final/time_{idx_time}/2foldcv_discovery_coefs.csv")

    coefs_replication_table = pd.DataFrame()
    i = 1
    for coef_fold in coefs_replication:
        coefs_replication_table[f"fold_{i}_coefs"] = coef_fold
        i += 1
        
    coefs_replication_table.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/res_100_times_final/time_{idx_time}/2foldcv_replication_coefs.csv")
    
if __name__ == "__main__":
    discovery_nonzero_mat_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery.npy'
    replication_nonzero_mat_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication.npy'
    discovery_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_removed_siblings.csv')
    replication_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_removed_siblings.csv')
    discovery_nonzero_indices = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery_nonzero_indices.csv')
    replication_nonzero_indices = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_nonzero_indices.csv')
    c_range_start = int(sys.argv[1])
    c_range_end = int(sys.argv[2])
    start_time_idx = int(sys.argv[3])
    end_time_idx = int(sys.argv[4])
    for time_idx in range(start_time_idx, end_time_idx):
        svm_wrapper(c_range_start, c_range_end, time_idx)