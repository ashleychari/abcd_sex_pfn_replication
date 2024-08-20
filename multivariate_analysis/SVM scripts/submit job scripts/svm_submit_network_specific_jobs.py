import os

for i in range(1, 18):
    os.system(f"qsub -l h_vmem=50G,s_vmem=50G svm_run_network_specific_models.sh {i}")