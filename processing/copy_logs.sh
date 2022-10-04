#!/bin/bash

mkdir -p fixed_logs/all_logs
cd fixed_logs/all_logs
cp -r ../uniform_name/* .
cp -r /home/ssujit/projects/def-ebrahimi/ssujit/jaxrl/fixed_logs/uniform_name/* .
cp -r /home/ssujit/projects/def-ebrahimi/ssujit/td3_bc_jax/fixed_logs/uniform_name/* .
cp -r /home/ssujit/projects/rrg-ebrahimi/ssujit/JaxCQL/fixed_logs/uniform_name/* .