import os

# Submit 1000 jobs
for i in range(1, 1001):
    os.system(f"sbatch svm_permutation_parallel.sh {i}")