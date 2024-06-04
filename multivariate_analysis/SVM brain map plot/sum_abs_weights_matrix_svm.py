import pandas as pd
import numpy as np

def create_abs_sum_matrix(matrix_filename, save_filename):
    # Load in haufe transformed weight matrix of shape (17, 59412)
    weight_brain_matrix = np.load(matrix_filename)
    # Make values absolute values
    abs_weight_brain_matrix = abs(weight_brain_matrix)
    # Sum along the rows so resulting matrix is of size (59412, )
    sum_abs_weight_brain_matrix = np.sum(abs_weight_brain_matrix, axis=0)
    # Save matrix
    np.save(save_filename, sum_abs_weight_brain_matrix)
    print("Job done!")

if __name__ == "__main__":
    create_abs_sum_matrix("discovery_haufe_transformed_100_runs_weights_final.npy", "abs_sum_weight_brain_mat_discovery_haufe_100_runs.npy")
    create_abs_sum_matrix("replication_haufe_transformed_100_runs_weights_final.npy", "abs_sum_weight_brain_mat_replication_haufe_100_runs.npy")
