#!/bin/bash

source ~/ENV/bin/activate

labels="mdl_baseline Baseline mdl_strat_freq1-1 obs1_gsteps1 mdl_strat_freq1-2 obs1_gsteps2"

xlim=1000000
regex=".*"
regex="mdl_str"
regex="mdl_"
fname="baseline"
tasks='.*'
palette_choice=contrast
logdir="logs/mdl"
add="none"
cols=4
bins=50000
yaxis="evaluation/average_returns"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1

logdir="logs/mdl"
regex=".*"
tasks='finetune'
fname="finetune"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1

logdir="logs/mdl"
regex=".*"
cols=3
tasks='online'
fname="online"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1

regex=".*"
tasks='mixed'
fname="mixed"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200 --baselines Dataset --plot_baseline 1

