#!/bin/bash

games="halfcheetah walker2d hopper"
# games="hopper walker2d"
# games="walker2d"

levels="mixed"
# levels="medium medium-expert"
levels="random medium medium-expert medium-replay"
algos="awac bc"
# algos="awac"
for game in $games
do
    for level in $levels
    do
        for algo in $algos
        do
            job_name="${game}"
            job_args="${game} ${level} ${algo}"
            mkdir -p out/${job_name}
            sbatch_args="--job-name $job_name"

            sbatch_args="${sbatch_args} --time=100:00"

            # run_cmd="scripts/exp.sh ${job_args}"
            run_cmd="scripts/mdl_exp.sh ${job_args}"

            # run_cmd="scripts/finetune_mdl_exp.sh ${job_args}"

            sbatch_cmd="sbatch ${sbatch_args} ${run_cmd}"
            cmd="$run_cmd"
            cmd="$sbatch_cmd"
            echo -e "${cmd}"
            ${cmd}
            sleep 1
        done
    done
done