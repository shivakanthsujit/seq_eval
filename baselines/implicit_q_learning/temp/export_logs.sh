#!/bin/bash

mkdir -p fixed_logs/export_logs
cd logs/mdl
cp -r **/*mdl_strat*/**/metrics.jsonl ../../fixed_logs/export_logs --parents
cp -r online-**/**/**/metrics.jsonl ../../fixed_logs/export_logs --parents
cp -r finetune-**/**/**/metrics.jsonl ../../fixed_logs/export_logs --parents
cd ../..
python temp/export_logs.py