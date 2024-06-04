import os

for i in range(1, 1001, 4):
    start_index = i
    end_index = start_index + 4
    os.system(f"qsub -l h_vmem=50G,s_vmem=50G svm_permutation_batched.sh {start_index} {end_index}")