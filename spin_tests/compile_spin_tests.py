import pandas as pd
import os

if __name__ == "__main__":
    spin_results_folder = "spin_test_check"
    results_dfs = []
    for results_file in os.listdir(spin_results_folder):
        # check extension
        if results_file.split(".")[1] == "csv":
            results_dfs.append(pd.read_csv(f"{spin_results_folder}/{results_file}"))
    
    all_results = pd.concat(results_dfs, axis=0)
    all_results.to_csv(f"{spin_results_folder}/spin_test_compiled_results.csv")
    print("Job Done!")