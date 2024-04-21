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

# Arielle's approach to the permutation 100 times


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

    
    # create empty dataframe to hold covariates
    # covariates_discovery = pd.DataFrame()
    # discovery_df = data_for_ridge[data_for_ridge['matched_group'] == 1]
    # meanFD_discovery = discovery_df['meanFD']
    # age_discovery = discovery_df['interview_age'].values/12

    # # Add to df
    # covariates_discovery['subject_id'] = discovery_df['subjectkey']
    # covariates_discovery['motion'] = meanFD_discovery
    # covariates_discovery['age'] = age_discovery

    # # create empty dataframe to hold covariates
    # covariates_replication = pd.DataFrame()
    # replication_df = data_for_ridge[data_for_ridge['matched_group'] == 2]
    # meanFD_replication= replication_df['meanFD']
    # age_replication = replication_df['interview_age'].values/12

    # # Add to df
    # covariates_replication['subject_id'] = replication_df['subjectkey']
    # covariates_replication['motion'] = meanFD_replication
    # covariates_replication['age'] = age_replication
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



def bootstrap_svm(num_rounds, discovery_matrix_filename, replication_matrix_filename, discovery_optimal_c, 
                  replication_optimal_c, discovery_data, replication_data):
    print("Running bootstrap code....")
    X_discovery = load_nonzero_mat(discovery_matrix_filename)
    X_replication = load_nonzero_mat(replication_matrix_filename)
    X  = np.vstack((X_discovery, X_replication))
    all_data = pd.concat([discovery_data, replication_data])
    covariates_discovery, covariates_replication = add_covariates(discovery_data, replication_data)
    covariates_data = pd.concat([covariates_discovery, covariates_replication])

    bootstrap_accuracies = []
    for i in range(num_rounds):
        fold_group_random = np.random.choice(all_data['matched_group'].values, all_data.shape[0], replace=False)
        A = np.argwhere(fold_group_random==1) # <- save an index of everywhere the fold group is 1
        B = np.argwhere(fold_group_random==2) # <- save an index of everywhere the fold group is 2

        X_discovery = X[A]
        y_discovery = all_data['sex'].values[A]
        y_discovery = [1 if y=="M" else -1 for y in y_discovery]
        X_replication = X[B]
        y_replication = all_data['sex'].values[B]
        y_replication = [1 if y=="M" else -1 for y in y_replication]

        covariates_data = covariates_data.drop(['subject_id', 'matched_group'], axis=1)
        nuisance_discovery = covariates_data.values[A]
        nuisance_replication = covariates_data.values[B]

        # Remove the unnecessary null dimension
        X_discovery = np.squeeze(X_discovery)
        y_discovery = np.squeeze(y_discovery)
        X_replication = np.squeeze(X_replication)
        y_replication = np.squeeze(y_replication)
        nuisance_discovery=np.squeeze(nuisance_discovery)
        nuisance_replication =np.squeeze(nuisance_replication)

        # Train on discovery, test on replication
        nuisance_model = LinearRegression()
        nuisance_model.fit(nuisance_discovery, X_discovery)
        X_discovery = X_discovery - nuisance_model.predict(nuisance_discovery)
        X_replication = X_replication - nuisance_model.predict(nuisance_replication)

        # Min Max Scale X 
        print("Scaling data....")
        scaler = MinMaxScaler()
        X_discovery = scaler.fit_transform(X_discovery)
        X_replication = scaler.fit_transform(X_replication)

        # Train on discovery, test on replication
        print("Running SVM...")
        model = svm.SVC(kernel='linear', C=discovery_optimal_c)
        model.fit(X_discovery, y_discovery)
        replication_predictions = model.predict(X_replication)
        replication_accuracy = accuracy_score(y_replication, replication_predictions)
        bootstrap_accuracies.append(replication_accuracy)

        # Train on replication, test on discovery
        X_discovery = X[A]
        X_replication = X[B]
       
        nuisance_discovery = covariates_data.values[A]
        nuisance_replication = covariates_data.values[B]

        # # Remove the unnecessary null dimension
        X_discovery = np.squeeze(X_discovery)
        y_discovery = np.squeeze(y_discovery)
        X_replication = np.squeeze(X_replication)
        y_replication = np.squeeze(y_replication)
        nuisance_discovery=np.squeeze(nuisance_discovery)
        nuisance_replication =np.squeeze(nuisance_replication)

        print("Fitting linear regression model on replication set....")
        nuisance_model = LinearRegression()
        nuisance_model.fit(nuisance_replication, X_replication)
        X_discovery = X_discovery - nuisance_model.predict(nuisance_discovery)
        X_replication = X_replication - nuisance_model.predict(nuisance_replication)

        # Min Max Scale X 
        print("Scaling....")
        scaler = MinMaxScaler()
        X_discovery = scaler.fit_transform(X_discovery)
        X_replication = scaler.fit_transform(X_replication)

        # Train on replication, test on discovery
        print("Running svm....")
        model = svm.SVC(kernel='linear', C=replication_optimal_c)
        model.fit(X_replication, y_replication)
        discovery_predictions = model.predict(X_discovery)
        discovery_accuracy = accuracy_score(y_discovery, discovery_predictions)
        bootstrap_accuracies.append(discovery_accuracy)

    bootstrap_df = pd.DataFrame()
    bootstrap_df['accuracies'] = bootstrap_accuracies
    bootstrap_df.to_csv(f"/cbica/projects/ash_pfn_sex_diff_abcd/results/multivariate_analysis/boostrap_{num_rounds}_accuracy_table.csv")

if __name__ == "__main__":
    
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

    bootstrap_svm(100, discovery_nonzero_mat_filename, replication_nonzero_mat_filename, 
                  discovery_optimal_c, replication_optimal_c, discovery_data_for_ridge, replication_data_for_ridge)
