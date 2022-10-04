#!/bin/bash

games="halfcheetah walker2d hopper"
# games="hopper walker2d"
# games="halfcheetah"

# levels="mixed"
levels="random medium"
levels="medium-expert"
levels="medium-replay"
for game in $games
do
    for level in $levels
    do
        job_name="${game}"
        job_args="${game} ${level} ${algo}"
        mkdir -p out/${job_name}

        sbatch_args="--job-name $job_name"

        # MDL Exp
        # sbatch_args="${sbatch_args} --time=2:00:00"
        # sbatch_args="${sbatch_args} --time=4:00:00"
        sbatch_args="${sbatch_args} --time=45:00"
        
        # run_cmd="scripts/exp.sh ${job_args}"
        run_cmd="scripts/mdl_exp.sh ${job_args}"

        # For finetuning, add 1.5 hours to each of the above
        # sbatch_args="${sbatch_args} --time=3:30:00"
        # sbatch_args="${sbatch_args} --time=5:30:00"
        # sbatch_args="${sbatch_args} --time=2:30:00"

        # run_cmd="scripts/finetune_mdl_exp.sh ${job_args}"
        
        sbatch_cmd="sbatch ${sbatch_args} ${run_cmd}"
        cmd="$run_cmd"
        cmd="$sbatch_cmd"
        echo -e "${cmd}"
        ${cmd}
        sleep 1
    done
done

# 31775206 hopper medium-expert cql
# 31775205 walker2d medium-expert cql
# 31775204 halfcheetah medium-expert cql

# 31776898 walker2d medium td3bc
# 31776897 halfcheetah medium td3bc

# 31775180 hopper medium-expert cql
# 31775179 walker2d medium-expert cql
# 31775178 halfcheetah medium-expert cql