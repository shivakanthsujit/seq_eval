# Bridging the Gap Between Offline and Online Reinforcement Learning Evaluation Methodologies

[Shivakanth Sujit](http://shivakanthsujit.github.io/), [Pedro Braga](https://phbraga.com/), [Jörg Bornschein](https://scholar.google.com/citations?user=X7kZFnoAAAAJ&hl=en), [Samira Ebrahimi Kahou](https://saebrahimi.github.io/)


[![arXiv](https://img.shields.io/badge/arXiv-2212.08131-b31b1b.svg)](https://arxiv.org/abs/2212.08131) [![pub](https://img.shields.io/badge/OfflineRL-NeurIPS2022-blue.svg)](https://openreview.net/pdf?id=lT4dOUtZYZ)

[Wesbite](https://shivakanthsujit.github.io/seq_eval/) | [Slides](https://drive.google.com/file/d/1tgbYR4W8bVCJxN-nYUjwwvw6rvIUEizx/view?usp=sharing) | [Poster](https://drive.google.com/file/d/1f6zkS-vRFi9YdBGR6RKgh8MfG6Jm5MtT/view?usp=sharing) | [Tweet](https://twitter.com/ShivaSujit/status/1598711848936800260)

This repo provides the code for the Seq Eval paper.

If you find our work useful, please cite us in your paper.

```
@article{sujit2022bridging,
        title={Bridging the Gap Between Offline and Online Reinforcement Learning Evaluation Methodologies},
        author={Shivakanth Sujit and Pedro Braga and Jorg Bornschein and Samira Ebrahimi Kahou},
        publisher = {arXiv},
        year = {2022},
        url = {https://arxiv.org/abs/2212.08131},
      }
```

## Overview

Reinforcement learning (RL) has shown great promise with algorithms learning in environments with large state and action spaces purely from scalar reward signals. A crucial challenge for current deep RL algorithms is that they require a tremendous amount of environment interactions for learning. This can be infeasible in situations where such interactions are expensive; such as in robotics. Offline RL algorithms try to address this issue by bootstrapping the learning process from existing logged data without needing to interact with the environment from the very beginning. While online RL algorithms are typically evaluated as a function of the number of environment interactions, there exists no single established protocol for evaluating offline RL methods. In this paper, we propose a sequential approach to evaluate offline RL algorithms as a function of the training set size and thus by their data efficiency. Sequential evaluation provides valuable insights into the data efficiency of the learning process and the robustness of algorithms to distribution changes in the dataset while also harmonizing the visualization of the offline and online learning phases. Our approach is generally applicable and easy to implement. We compare several existing offline RL algorithms using this approach and present insights from a variety of tasks and offline datasets.

## Usage of Repo
The codebases for each baseline are present in a separate folder inside `baselines`. Each folder contains a `scripts` folder that contains the files to reproduce the results in the paper.

### Experiment running
To start the runs for each baseline, you can use the `run.sh` file in the `scripts` folder. It will start the runs for each environment in the benchmark.

In the run script you can choose which experiment to run.

Each codebase contains the instructions required to set it up from the original authors. No additional setup is required to run the code.

### Plotting

Each `scripts` folder has a `plot.sh` that can be run to produce the main plots from the paper. There is a `results` folder that contains the figures from the paper as well as data that can be used to recreate the figure. There are also CSVs with final scores of each environment and baseline presented in the paper.

## Acknowledgements

* The IQL baseline is modified from the [here](https://github.com/ikostrikov/implicit_q_learning).
* The CQL baseline is modified from [here](https://github.com/young-geng/JaxCQL).
* The TD3+BC baseline is modified from [here](https://github.com/ethanluoyc/td3_bc_jax).
* The AWAC and BC baseline is modified from [here](https://github.com/ikostrikov/jaxrl).