#!/bin/bash
#SBATCH --time=2:15:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --job-name=hopper
#SBATCH --array=1-3
#SBATCH --output=out/%x/%A_%a.out
#SBATCH --error=out/%x/%A_%a.err
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=shivakanth.sujit@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

source ~/jax_env/bin/activate

module load cuda/11.1

game=${1:-"halfcheetah"}
level=${2:-"medium"}

logdir="logs"

# logdir="logs/testing"
# SLURM_ARRAY_TASK_ID=1

seed=${SLURM_ARRAY_TASK_ID}
# test_args="--eval_freq 10000"
# mdl_strat="freq50000-1"

mdl_strat="freq1-1"
test_args="${test_args} --mdl_strat ${mdl_strat}"
ID="mdl_strat_${mdl_strat}"

# max_steps=100000
# ID="${ID}_100k"
max_timesteps=500000
ID="${ID}_500k"
test_args="${test_args} --max_timesteps ${max_timesteps}"

log_msg="${game}/${ID} is being logged to ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out\n"
echo -e "$(date) ${log_msg}"
echo -e "$(date) ${log_msg}" >> out/${game}/job_logs.txt

python main_finetune_mdl.py --env=${game}-${level}-v2 --logdir=${logdir} --id=${ID} --seed=${seed} ${test_args}

echo -e "$(date) ${log_msg}"
echo -e "$(date) ${log_msg}" >> out/${game}/completed_job_logs.txt