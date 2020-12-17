#!/bin/bash
#SBATCH -p priority                          #partition
#SBATCH -t 0-6:00                        # time days-hr:min
#SBATCH -c 8                              # number of cores
#SBATCH --mem=40G                          # memory per job (all cores), GB
#SBATCH -o /n/data1/hms/cellbio/sander/bo/julia/slurm/%j.out                         # out file, %j would be your job id
#SBATCH -e /n/data1/hms/cellbio/sander/bo/julia/slurm/%j.err                         # error file, $j would be your job id
source ~/.bash_profile
julia="/n/data1/hms/cellbio/sander/bo/julia/installation/julia-1.5.3/bin/julia"

cd /n/data1/hms/cellbio/sander/bo/julia/cellbox-julia-dev-6338
${julia} -p 8 "${1}"

#### 3-5 min small jobs
