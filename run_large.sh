#!/bin/bash
#SBATCH -p short                          #partition
#SBATCH -t 0-12:00                        # time days-hr:min
#SBATCH -c 8                              # number of cores
#SBATCH --mem=40G                          # memory per job (all cores), GB
#SBATCH -o /n/data1/hms/cellbio/sander/bo/julia/slurm/%j.out                         # out file, %j would be your job id
#SBATCH -e /n/data1/hms/cellbio/sander/bo/julia/slurm/%j.err                         # error file, $j would be your job id
source ~/.bash_profile
julia="/n/data1/hms/cellbio/sander/bo/julia/installation/julia-1.5.3/bin/julia"

cd /n/data1/hms/cellbio/sander/bo/julia/cellbox-julia-dev-6338
${julia} -p 8 ${1}

# 20-30 min large jobs
# Task A - Part I - 48 * 20 min
# sbatch run_large.sh final_train5_o2_TaskA/net5_50.jl
# Task A - Part II - 48 * 20 min
# sbatch run_large.sh final_train5_o2_TaskA/net10_20.jl
# Task A - Part III - 24 * 40 min
# sbatch run_large.sh final_train5_o2_TaskA/net100.jl

# Task B - Part I - 96 * 10 min
# sbatch run_large.sh final_train5_o2_TaskB/ts1_40.jl
# Task B - Part II - 96 * 10 min
# sbatch run_large.sh final_train5_o2_TaskB/ts5_10.jl
