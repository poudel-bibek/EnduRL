## EnduRL: Enhancing Safety, Stability, and Efficiency of Mixed Traffic Under Real-World Perturbations Via Reinforcement Learning
[arXiv](https://arxiv.org/abs/2311.12261)

### Appendix section

------
This work was done on top of [FLOW](https://github.com/flow-project/flow) framework obtained on Jan 3, 2023.

### Installation instructions 

Developed and tested on Ubuntu 18.04, Python 3.7.3

- Install [Anaconda](https://www.anaconda.com/)
- Setup [SUMO versio 1.15.0](https://github.com/eclipse-sumo/sumo/releases/tag/v1_15_0)
- Clone this repository
- Use the following commands

```
conda env create -f environment.yml
conda activate flow
python setup.py develop
pip install -U pip setuptools
pip install -r requirements.txt
```

### Part 1: Training
```
python train.py --exp_config singleagent_ring
python train.py singleagent_bottleneck
python train.py intersection
```

To view tensorboard while training: 
```
tensorboard --logdir=~/ray_results/
```

## Part 2: Generate rollouts from trained RL agent or using Classic RVs (Heuristic and Model based) and save as csv files.
### RL agents:
Replace the method name to be one of: ours, wu

```
python test_rllib.py [Location of trained policy] [checkpoint number] --method wu --gen_emission --num_rollouts 10 --shock --render --length 260
```

### Classic RVs (Heuristic and Model based):
For all (replace the method_name to be one of: bcm, lacc, piws, fs, idm)
```
python classic.py --method [method_name] --render --length 260 --num_rollouts [no_of_rollouts] --shock --gen_emission
```

For stability tests where just the leader adds perturbations, include --stability to the lines above

## Part 3: Evaluate the generated rollouts

To evaluate the generated rollouts into Safety, Efficiency and Stability metrics:
Replace the method name to be one of: bcm, idm, fs, piws, lacc, wu, ours

```
python eval_metrics.py --method [method_name] --num_rollouts [no_of_rollouts]
```

To add plots to the metrics, include --save_plots

For Stability plots
```
python eval_plots.py --method [method_name]
```

-------------------------------------


Data (including published experiment rollouts and videos): [HuggingFace](https://huggingface.co/datasets/matrix-multiply/EnduRL_data/tree/main)

Trained CSC Models: [HuggingFace](https://huggingface.co/matrix-multiply/Congestion_Stage_Classifier/tree/main)


------------
Locations: 

./Ours/Trained_policies/Last_good/weak_accept_policy/PPO_DensityAwareRlEnv-v0_719f478a_2022-06-05_13-36-42okip6tqy 18

./Wu_et_al/Trained_policies/PPO_WaveAttenuationPOEnv-v0_25b5cb6e_2022-01-26_10-58-12e9f4i3ao 50 

Requirements have been modified 

---------------------
To generate rollouts without shock: 
python classic.py --method bcm --render --length 220 --num_rollouts 20 --gen_emission

for LACC, Shock start and end times are 1140 and 1500 respectively

-------
## Cite

```

@article{poudel2024endurl,
  title={EnduRL: Enhancing Safety, Stability, and Efficiency of Mixed Traffic Under Real-World Perturbations Via Reinforcement Learning},
  author={Poudel, Bibek and Li, Weizi and Heaslip, Kevin},
  journal={arXiv preprint arXiv:2311.12261},
  year={2024}
}

```
## License

[MIT](https://choosealicense.com/licenses/mit/)