#!/bin/bash 
#SBATCH -J mdt_calvin_hk
#SBATCH -p gpux
#SBATCH -n 4       # Number of tasks
#SBATCH -c 16      # Number of cores per task
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time 72:00:00
#SBATCH --comment=LightVLA
#SBATCH -o /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt/jobs/mdt_calvin_hk/std.out.%j
#SBATCH -e /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt/jobs/mdt_calvin_hk/std.err.%j


#########################################################
### Slurm scripts for Sugon Portal_5.0 of BASE
### Version 1.0    |  2019-09-09  |  created by Zhang Guoliang
### Version 1.0.1  |  2020-11-24  |  modified by Zhang Guoliang
#########################################################

### Get parameters from GUI

MIDFILE_DIR=/public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt/jobs/mdt_calvin_hk/.portal
source $MIDFILE_DIR/job_portal.var
source $MIDFILE_DIR/job_interface.var

### Set basic var   ### MARK_slurm2pbs

WORK_DIR=/public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt/jobs/mdt_calvin_hk
JOB_NAME=mdt_calvin_hk
JOBID=$SLURM_JOB_ID
NP=$SLURM_NPROCS
NNODE=`srun hostname | sort | uniq | wc -l`

LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma 

# Create the directory if it doesn't exist
mkdir -p $WORK_DIR

# Generate host file
srun hostname | sort | uniq -c | awk '{print $2":"$1}' > $HOST_FILE

# Log basic job information
echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n" | tee -a $LOG_FILE 

# Run the application
echo "Running the application..." | tee -a $LOG_FILE

# Change to the project directory
cd /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt

# Properly activate conda environment
# Use the full path to conda and source the conda.sh file
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mdt_env

# Set environment variables
export TORCH_USE_CUDA_DSA=1

# Run the training script
srun python mdt/training.py

echo "The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE"
