#!/bin/bash

source ~/ENV/bin/activate

labels="baseline Baseline mdl_strat_freq1-1 obs1_gsteps1 mdl_strat_freq1-2 obs1_gsteps2"

xlim=1000000
regex=".*"
regex="mdl"
fname="baseline"
tasks='.*'
palette_choice=contrast
logdir="logs"
add="none"
cols=3
bins=10000
yaxis="average_normalizd_return"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

regex=".*"
tasks='finetune'
fname="finetune"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

regex=".*"
tasks='online'
fname="online"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

regex=".*"
tasks='mixed'
fname="mixed"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${bins} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200
