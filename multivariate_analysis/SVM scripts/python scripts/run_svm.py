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

# Arielle's approach to 2fold cv with discovery and replication
# as one of the folds


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


def get_optimal_C(matrix_filename, discovery_data, replication_data, nonzero_indices, C_range, discovery_replication="discovery"):
    features_nonzero_matrix = load_nonzero_mat(matrix_filename)
    data_for_ridge = pd.concat([discovery_data, replication_data], axis=0)
    covariates_discovery, covariates_replication = add_covariates(discovery_data, replication_data)
    if discovery_replication == "discovery":
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

    if discovery_replication == "discovery":
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
    return cs[best_c_ind]

def run_2fold_svm(discovery_matrix_filename, replication_matrix_filename, discovery_nonzero_indices, replication_nonzero_indices, 
                  discovery_optimal_c, replication_optimal_c, discovery_data, replication_data):
    print("Running 2 fold svm...")
    # Setup Xs, Ys, and covariates
    X_discovery = load_nonzero_mat(discovery_matrix_filename)
    X_replication = load_nonzero_mat(replication_matrix_filename)
    discovery_covariates, replication_covariates = add_covariates(discovery_data, replication_data)
    discovery_covariates = discovery_covariates.drop(['subject_id'], axis=1)
    replication_covariates = replication_covariates.drop(['subject_id'], axis=1)
    y_discovery = discovery_data['sex']
    y_discovery = np.array([1 if y=="M" else -1 for y in y_discovery])
    y_replication = replication_data['sex']
    y_replication = np.array([1 if y=="M" else -1 for y in y_replication])

    # Remove the unnecessary null dimension
    X_discovery = np.squeeze(X_discovery)
    y_discovery = np.squeeze(y_discovery)
    X_replication = np.squeeze(X_replication)
    y_replication = np.squeeze(y_replication)
    discovery_covariates=np.squeeze(discovery_covariates)
    replication_covariates =np.squeeze(replication_covariates)

    # Train on discovery, predict on replication
    # Fit on train data and predict on train and test
    print("removing nuisance....")
    nuisance_model = LinearRegression()
    nuisance_model.fit(discovery_covariates, X_discovery)
    X_discovery = X_discovery - nuisance_model.predict(discovery_covariates)
    X_replication = X_replication - nuisance_model.predict(replication_covariates)

    # Min Max Scale X 
    print("scaling....")
    scaler = MinMaxScaler()
    X_discovery = scaler.fit_transform(X_discovery)
    X_replication = scaler.fit_transform(X_replication)

    print("Fitting svm...")
    model = svm.SVC(kernel="linear", random_state=42, C=discovery_optimal_c)
    model.fit(X_discovery, y_discovery)
    # Predict on replication set
    y_pred_replication = model.predict(X_replication)
    accuracy_replication = accuracy_score(y_replication, y_pred_replication)
    auc_replication = roc_auc_score(y_replication, y_pred_replication)
    metrics_table_replication = pd.DataFrame()
    metrics_table_replication['accuracy'] = [accuracy_replication]
    metrics_table_replication['auc'] = [auc_replication]
    metrics_table_replication.to_csv("/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/discovery_train_replication_test_metrics_table.csv")
    fpr, tpr, thresholds = metrics.roc_curve(y_replication, y_pred_replication)
    roc_auc = metrics.auc(fpr, tpr)
    display1 = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # RocCurveDisplay.from_estimator(model, X_discovery, y_discovery)
    display1.plot()
    plt.savefig("/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/discovery_train_replication_test_roc_curve.png")

    
    discovery_model_coefs = pd.DataFrame()
    #discovery_model_coefs['nonzero_indices'] = discovery_nonzero_indices
    discovery_model_coefs['coefs'] = list(model.coef_[0])
    discovery_model_coefs.to_csv("/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/discovery_train_replication_test_coefs.csv")


    # Train on replication, predict on discovery
    # Fit on train data and predict on train and test
    X_discovery = load_nonzero_mat(discovery_matrix_filename)
    X_replication = load_nonzero_mat(replication_matrix_filename)
    nuisance_model = LinearRegression()
    nuisance_model.fit(replication_covariates, X_replication)
    X_discovery = X_discovery - nuisance_model.predict(discovery_covariates)
    X_replication = X_replication - nuisance_model.predict(replication_covariates)

    # Min Max Scale X 
    scaler = MinMaxScaler()
    X_discovery = scaler.fit_transform(X_discovery)
    X_replication = scaler.fit_transform(X_replication)

    model = svm.SVC(kernel="linear", random_state=42, C=replication_optimal_c)
    model.fit(X_replication, y_replication)
    # Predict on discovery set
    y_pred_discovery = model.predict(X_discovery)
    accuracy_discovery = accuracy_score(y_discovery, y_pred_discovery)
    auc_discovery = roc_auc_score(y_discovery, y_pred_discovery)
    metrics_table_discovery = pd.DataFrame()
    metrics_table_discovery['accuracy'] = [accuracy_discovery]
    metrics_table_discovery['auc'] = [auc_discovery]
    metrics_table_discovery.to_csv("/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/discovery_test_replication_train_metrics_table.csv")
    fpr, tpr, thresholds = metrics.roc_curve(y_discovery, y_pred_discovery)
    roc_auc = metrics.auc(fpr, tpr)
    display2 = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    # RocCurveDisplay.from_estimator(model, X_discovery, y_discovery)
    display2.plot()
    plt.savefig("/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/discovery_test_replication_train_roc_curve.png")


    replication_model_coefs = pd.DataFrame()
    #replication_model_coefs['nonzero_indices'] = replication_nonzero_indices
    replication_model_coefs['coef'] = list(model.coef_[0])
    replication_model_coefs.to_csv("/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/discovery_test_replication_train_coefs.csv")


if __name__ == "__main__":
    # Things to do
    # 1. Hardcode in filepaths
    # 2. Make the type of test choosable
    # 3. Create null tests
    
    discovery_nonzero_mat_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery.npy'
    replication_nonzero_mat_filename = '/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication.npy'
    discovery_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/discovery_sample_removed_siblings.csv')
    replication_data_for_ridge = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/dropbox/replication_sample_removed_siblings.csv')
    discovery_nonzero_indices = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_discovery_nonzero_indices.csv')
    replication_nonzero_indices = pd.read_csv('/cbica/projects/ash_pfn_sex_diff_abcd/results/AtlasLoading_All_RemoveZero_replication_nonzero_indices.csv')
    c_range_start = int(sys.argv[1])
    c_range_end = int(sys.argv[2])

    c_range = range(c_range_start, c_range_end)
    c_values = [2**i for i in c_range]
    discovery_optimal_c = get_optimal_C(discovery_nonzero_mat_filename, discovery_data_for_ridge, replication_data_for_ridge, discovery_nonzero_indices, c_values, "discovery")
    replication_optimal_c = get_optimal_C(replication_nonzero_mat_filename, discovery_data_for_ridge, replication_data_for_ridge, replication_nonzero_indices, c_values, "replication")

    run_2fold_svm(discovery_nonzero_mat_filename, replication_nonzero_mat_filename, discovery_nonzero_indices, 
                      replication_nonzero_indices, discovery_optimal_c, 
                  replication_optimal_c, discovery_data_for_ridge, replication_data_for_ridge)
    
    
