#!/bin/bash

games="halfcheetah walker2d hopper"
# games="hopper walker2d"
# games="halfcheetah"

levels="medium"

for game in $games
do
    for level in $levels
    do
        job_name="${game}"
        job_args="${game} ${level}"
        sbatch_args="--job-name $job_name"

        mkdir -p out/${job_name}
        
        sbatch_args="${sbatch_args} --time=1:30:00"
        run_cmd="scripts/online_exp.sh ${job_args}"
        
        sbatch_cmd="sbatch ${sbatch_args} ${run_cmd}"
        cmd="$run_cmd"
        cmd="$sbatch_cmd"
        echo -e "${cmd}"
        ${cmd}
        sleep 1
    done
done