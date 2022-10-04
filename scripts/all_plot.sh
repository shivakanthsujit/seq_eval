#!/bin/bash

source ~/ENV/bin/activate

save_runs=True

labels="TD3BC_mdl_strat_freq1-1 TD3+BC awac_mdl_strat_freq1-1 AWAC bc_mdl_strat_freq1-1 BC cql_mdl_strat_freq1-1 CQL iql_mdl_strat_freq1-1 IQL"
labels="${labels} TD3BC_mdl_strat_freq1-2 TD3+BC awac_mdl_strat_freq1-2 AWAC bc_mdl_strat_freq1-2 BC cql_mdl_strat_freq1-2 CQL iql_mdl_strat_freq1-2 IQL"

regex="mdl_strat_freq1-1"
fname="fig1"
tasks="(^half|^hopper|^walker).*(random|medium)"
palette_choice=contrast
logdir="fixed_logs/all_logs"
add="none"
cols=4
bins=50000
yaxis="eval_returns"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}

tasks='(^half|^hopper).*(random|medium)'
cols=4
regex="mdl_strat_freq1-1"
fname="d4rl"
other_args="--legend_options ncol 1 loc center_right"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 ${other_args} --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}

tasks='.*'
cols=3
regex="mdl_strat_freq1-2"
fname="fig2"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}

labels="${labels} TD3BC_mdl_strat_freq1-1_500k TD3+BC awac_mdl_strat_freq1-1_500k AWAC bc_mdl_strat_freq1-1_500k BC cql_mdl_strat_freq1-1_500k CQL iql_mdl_strat_freq1-1_500k IQL"
tasks='finetune-halfcheetah'
regex=".*"
cols=4
fname="finetune_main"
other_args="--legend_options ncol 1 loc center_right"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 ${other_args} --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}

labels="${labels} TD3BC_mdl_strat_freq1-1_500k TD3+BC awac_mdl_strat_freq1-1_500k AWAC bc_mdl_strat_freq1-1_500k BC cql_mdl_strat_freq1-1_500k CQL iql_mdl_strat_freq1-1_500k IQL"
tasks='finetune'
regex="500k$"
cols=4
fname="finetune"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}

tasks='online'
labels="${labels} TD3BC_baseline_2M TD3+BC awac_baseline_2M AWAC bc_baseline_2M BC cql_baseline_2M CQL iql_baseline_2M IQL"
cols=3
regex=".*"
fname="online"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}

tasks='mixed'
labels="${labels} TD3BC_mdl_strat_freq1-1_slidingwindow TD3+BC awac_mdl_strat_freq1-1_slidingwindow AWAC bc_mdl_strat_freq1-1_slidingwindow BC cql_mdl_strat_freq1-1_slidingwindow CQL iql_mdl_strat_freq1-1_slidingwindow IQL"
cols=3
regex="slidingwindow"
fname="mixed"
other_args="--legend_options ncol 1 loc center_right"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Samples --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 ${other_args} --baselines Dataset --plot_baseline 1 --save_runs ${save_runs}
