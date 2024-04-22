import pandas as pd
import numpy as np

def create_abs_sum_matrix(matrix_filename, save_filename):
    weight_brain_matrix = np.load(matrix_filename)
    abs_weight_brain_matrix = abs(weight_brain_matrix)
    sum_abs_weight_brain_matrix = np.sum(abs_weight_brain_matrix, axis=0)
    np.save(save_filename, sum_abs_weight_brain_matrix)
    print("Job done!")

if __name__ == "__main__":
    create_abs_sum_matrix("w_brain_sex_matrix_one_run.npy", "abs_sum_weight_brain_mat.npy")
    create_abs_sum_matrix("w_brain_sex_matrix_one_run_replication.npy", "abs_sum_weight_brain_mat_replication.npy")
