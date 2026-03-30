# Two-Time-Scale Learning Dynamics: A Population View of Neural Network Training

[![Paper](https://img.shields.io/badge/arXiv-2603.19808-b31b1b.svg)](https://arxiv.org/abs/2603.19808)

This repository contains the implementation of the experiments detailed in our paper on applying Population-Based Training (PBT) to Deep Q-Networks (DQN), illustrating the two-time-scale learning dynamics. 

## Overview
We explore the dynamic evolution of hyperparameters (`learning rate`, `epsilon decay`, and `batch size`) during the training of a DQN agent on the `CartPole-v1` environment. The codebase is designed to track hyperparameter distributions and evaluate the impact of fitness evaluation windows (deque sizes).

### Key Experiments
1. **Deque Size Performance:** Analyses how the size of the moving average window for fitness evaluation impacts the evolutionary trajectory (`plot_deque_performance`).
2. **Hyperparameter Distribution Evolution:** Visualizes the 3D evolution and density of hyperparameters across generations (`plot_3d_evolution`).

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/hyesungim/pbt-dqn-experiments.git](https://github.com/hyesungim/two-time-scale-learning.git)
cd two-time-scale-learning
pip install -r requirements.txt