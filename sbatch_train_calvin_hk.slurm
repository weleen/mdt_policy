#!/bin/bash 
#SBATCH -J mdt_calvin_hk
#SBATCH -p gpux
#SBATCH -n 4       # Number of tasks
#SBATCH -c 16  # Number of cores per task
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time 72:00:00
#SBATCH --comment=LightVLA
###SBATCH -o std.out.%j
###SBATCH -e std.err.%j
#SBATCH -o /public/home/group_xudong/yimingwu/BASE/mdt_calvin_hk/std.out.%j
#SBATCH -e /public/home/group_xudong/yimingwu/BASE/mdt_calvin_hk/std.err.%j


#########################################################
### Slurm scripts for Sugon Portal_5.0 of BASE
### Version 1.0    |  2019-09-09  |  created by Zhang Guoliang
### Version 1.0.1  |  2020-11-24  |  modified by Zhang Guoliang
#########################################################

### Get parameters from GUI

MIDFILE_DIR=/public/home/group_xudong/yimingwu/BASE/mdt_calvin_hk/.portal
source $MIDFILE_DIR/job_portal.var
source $MIDFILE_DIR/job_interface.var

### Set basic var   ### MARK_slurm2pbs

JOBID=$SLURM_JOB_ID                                  ### slurm2pbs
NP=$SLURM_NPROCS                                     ### slurm2pbs
NNODE=`srun hostname | sort | uniq | wc -l`          ### slurm2pbs

LOG_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}.log
HOST_FILE=$WORK_DIR/job_${JOB_NAME}_${JOBID}_${NP}c_${NNODE}n.ma 

srun hostname | sort | uniq -c |awk '{print $2":"$1}' > $HOST_FILE  ### slurm2pbs

### Write basic job infomations

echo -e "The start time is: `date +"%Y-%m-%d %H:%M:%S"` \n" | tee -a $LOG_FILE 
echo -e "My job ID is: $JOBID \n" | tee -a $LOG_FILE  
echo -e "The total cores is: $NP \n" | tee -a $LOG_FILE 
echo -e "The hosts is: \n" | tee -a $LOG_FILE
cat $HOST_FILE | tee -a $LOG_FILE
echo -e "\n"  | tee -a $LOG_FILE 

### Run APP

  # MARK_CMD  #Don't delete this line!!!
#!/bin/bash
## job script created by Gridview Jobmanager.

# Activate the virtualenv / conda environment
cd /public/home/group_xudong/yimingwu/project/MyProjects/LightVLA/examples/mdt

export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

python mdt/training.py

  # MARK_BASH #Don't delete this line!!!


echo The end time is: `date +"%Y-%m-%d %H:%M:%S"` | tee -a $LOG_FILE
