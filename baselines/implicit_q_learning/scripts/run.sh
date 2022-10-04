#!/bin/bash

games="halfcheetah walker2d hopper"
# games="hopper walker2d"
# games="halfcheetah"

# levels="mixed"
levels="random medium"
levels="medium-expert"
# levels="medium-replay"
for game in $games
do
    for level in $levels
    do
        job_name="${game}"
        job_args="${game} ${level}"
        sbatch_args="--job-name $job_name"
        mkdir -p out/${job_name}

        # MDL Exp
        # sbatch_args="${sbatch_args} --time=60:00"
        sbatch_args="${sbatch_args} --time=2:00:00"
        # sbatch_args="${sbatch_args} --time=30:00"

        # run_cmd="scripts/exp.sh ${job_args}"
        run_cmd="scripts/mdl_exp.sh ${job_args}"
        
        # Finetune Experiments. Add 30 mins for 500k steps
        # sbatch_args="${sbatch_args} --time=80:00"
        # sbatch_args="${sbatch_args} --time=2:30:00"
        # sbatch_args="${sbatch_args} --time=60:00"
        
        # run_cmd="scripts/finetune_mdl_exp.sh ${job_args}"
        
        sbatch_cmd="sbatch ${sbatch_args} ${run_cmd}"
        cmd="$run_cmd"
        cmd="$sbatch_cmd"
        echo -e "${cmd}"
        ${cmd}
        sleep 1
    done
done