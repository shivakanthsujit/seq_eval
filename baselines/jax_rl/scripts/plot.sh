#!/bin/bash

source ~/ENV/bin/activate

labels="baseline Baseline awac_mdl_strat_freq1-1 awac_obs1_gsteps1 awac_mdl_strat_freq1-2 awac_obs1_gsteps2"
labels="bc_mdl_strat_freq1-1 bc_obs1_gsteps1 bc_mdl_strat_freq1-2 bc_obs1_gsteps2"

xlim=1000000
regex="mdl"
fname="baseline"
tasks='.*'
palette_choice=contrast
logdir="logs"
add="none"
cols=4
bins=10000
yaxis="evaluation/average_returns"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

logdir="logs"
regex=".*"
tasks='finetune'
fname="finetune"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

cols=3
logdir="logs"
regex=".*"
tasks='online'
fname="online"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

regex=".*"
tasks='mixed'
fname="mixed"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200
