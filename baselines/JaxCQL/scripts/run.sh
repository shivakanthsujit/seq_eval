#!/bin/bash

games="halfcheetah walker2d hopper"
# games="hopper walker2d"
# games="hopper"

# levels="mixed"
# levels="random medium"
levels="random"
# levels="medium-expert"
levels="medium-replay"
for game in $games
do
    for level in $levels
    do
        job_name="${game}"
        job_args="${game} ${level}"
        sbatch_args="--job-name $job_name"
        mkdir -p out/${job_name}

        # * MDL Exp
        sbatch_args="${sbatch_args} --time=1:30:00"
        # sbatch_args="${sbatch_args} --time=2:00:00"
        sbatch_args="${sbatch_args} --time=50:00"
        
        # run_cmd="scripts/exp.sh ${job_args}"
        run_cmd="scripts/mdl_exp.sh ${job_args}"

        # * Finetuning Exps: Add 45 mins for finetuning
        # sbatch_args="${sbatch_args} --time=2:15:00"
        # sbatch_args="${sbatch_args} --time=2:45:00"
        # sbatch_args="${sbatch_args} --time=1:30:00"

        # run_cmd="scripts/finetune_mdl_exp.sh ${job_args}"
        
        sbatch_cmd="sbatch ${sbatch_args} ${run_cmd}"
        cmd="$run_cmd"
        cmd="$sbatch_cmd"
        echo -e "${cmd}"
        ${cmd}
        sleep 1
    done
done