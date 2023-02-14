Created on top of snapshot of FLOW code obtained on Jan 3, 2023

## Part 1: Generate rollouts
To evaluate Wu et al.:

To evaluate ours:  
```
python test_rllib.py [Location of trained policy] [checkpoint number] --gen_emission --num_rollouts 10 --render_mode no_render
```
To generate rollouts from traditional models and save to csv files:

```
python classic.py --method [method_name] --render --length 260 --num_rollouts [no_of_rollouts] --shock --gen_emission
```

For stability tests where just the leader adds perturbations, include --stability to the line above

## Part 2: Evaluate the generated rollouts

To evaluate the generated rollouts into Safety, Efficiency and Stability metrics:
```
python eval_metrics.py --method [method_name] --num_rollouts [no_of_rollouts]
```

To add plots to the metrics, include --save_plots

For Stability plots
```
python eval_plots.py --method [method_name]
```

-------------------------------------


## License

[MIT](https://choosealicense.com/licenses/mit/)

------------
Locations: 

./Ours/Trained_policies/Last_good/weak_accept_policy/PPO_DensityAwareRlEnv-v0_719f478a_2022-06-05_13-36-42okip6tqy 18

./Wu_et_al/Trained_policies/trained_here/PPO_WaveAttenuationPOEnv-v0_e2342e4c_2023-01-09_13-29-073it85esn 46

./Wu_et_al/Trained_policies/from_flow_code 200

Requirements have been modified 
