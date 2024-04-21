from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics


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


def run_network_specific_svm(matrix_filename, discovery_data, replication_data, matched_group, C_range):
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
    nuisance_model = LinearRegression()
    nuisance_model.fit(covariates, X)
    X = X - nuisance_model.predict(covariates)

    # Min Max Scale X 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if matched_group == "discovery":
        sex = data_for_ridge[data_for_ridge['matched_group'] == 1]['sex'].values
    else:
        sex = data_for_ridge[data_for_ridge['matched_group'] == 2]['sex'].values
    y = np.array([1 if i == "M" else -1 for i in sex]) # set Male=1, female=-1

    cs = []
    average_accuracies = []
    average_aucs = []
    for i in C_range:
        cs.append(i)
        fold_accuracies = []
        fold_aucs = []
        k_folds = KFold(n_splits=2, shuffle=True, random_state=42)
        for train_indices, test_indices in k_folds.split(X_scaled):
            X_train = X_scaled[train_indices]
            X_test = X_scaled[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            print("Fitting svm....")
            clf = svm.SVC(kernel='linear', C=i, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_aucs.append(roc_auc_score(y_test, y_pred))

    average_accuracies.append(np.mean(fold_accuracies))
    average_aucs.append(np.mean(fold_aucs))

    # Identify optimal c based on auc
    best_c_ind = np.argmax(average_aucs)
    best_c = cs[best_c_ind]

    # Repeat 2 fold cross validation
    coefs = []
    accuracies = []
    aucs = []
    fprs = []
    tprs = []
    k_folds = KFold(n_splits=2, shuffle=True, random_state=42)
    for train_indices, test_indices in k_folds.split(X_scaled):
        X_train = X_scaled[train_indices]
        X_test = X_scaled[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        print("Fitting svm....")
        clf = svm.SVC(kernel='linear', C=best_c, random_state=42)
        clf.fit(X_train, y_train)
        coefs.append(list(clf.coef_[0]))
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_pred))
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        fprs.append(fpr)
        tprs.append(tpr)

    normalized_coefs = []
    for coef in coefs:
        coef = coef / np.linalg.norm(coef)
        normalized_coefs.append(coef)

    return np.mean(accuracies), np.mean(aucs), np.mean(fpr), np.mean(tpr), normalized_coefs


if __name__ == "__main__":
    discovery_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_removed_siblings.csv')
    replication_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_removed_siblings.csv')
    matrices_folder = "/cbica/projects/ash_pfn_sex_diff_abcd/results/network_specific_matrices/"
    c_range_start = int(sys.argv[1])
    c_range_end = int(sys.argv[2])

    c_range = range(c_range_start, c_range_end)
    c_values = [2**i for i in c_range]
    for i in range(17):
        discovery_matrix_filename = f"{matrices_folder}/discovery_network_matrix_nonzero_{i}.npy"
        replication_matrix_filename = f"{matrices_folder}/replication_network_matrix_nonzero_{i}.npy"
        discovery_accuracies, discovery_aucs, discovery_fprs, discovery_tprs, discovery_coefs = run_network_specific_svm(discovery_matrix_filename, discovery_data_for_ridge, replication_data_for_ridge, "discovery", c_values)
        replication_accuracies, replication_aucs, replication_fprs, replication_tprs, replication_coefs = run_network_specific_svm(replication_matrix_filename, discovery_data_for_ridge, replication_data_for_ridge, "replication", c_values)
        metrics_table = pd.DataFrame()
        metrics_table['set'] = ['discovery', 'replication']
        metrics_table['accuracy'] = [discovery_accuracies, replication_accuracies]
        metrics_table['auc'] = [discovery_aucs, replication_aucs]
        metrics_table['tpr'] = [discovery_tprs, replication_tprs]
        metrics_table['fpr'] = [discovery_fprs, replication_fprs]
        metrics_table.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/network_specific_models/2foldcv_metrics_table_network_{i}.csv")

        coefs_discovery_table = pd.DataFrame()
        j = 1
        for coef_fold in discovery_coefs:
            coefs_discovery_table[f"fold_{j}_coefs"] = coef_fold
            j += 1
        coefs_discovery_table.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/network_specific_models/2foldcv_discovery_coefs_network_{i}.csv")

        coefs_replication_table = pd.DataFrame()
        j = 1
        for coef_fold in replication_coefs:
            coefs_replication_table[f"fold_{j}_coefs"] = coef_fold
            j += 1

        coefs_replication_table.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/network_specific_models/2foldcv_replication_coefs_network_{i}.csv")
        

        
    
        
