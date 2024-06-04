import os

for i in range(1, 101, 4):
    start_index = i
    end_index = start_index + 4
    os.system(f"qsub -l h_vmem=50G,s_vmem=50G svm_resmulti_times_parallel_batched.sh {start_index} {end_index}")