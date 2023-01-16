Created on top of snapshot of FLOW code obtained on Jan 3, 2023


To evaluate Wu et al.:

To evaluate ours:  
```
python test_rllib.py ~/ray_results/density_aware_rl/PPO_DensityAwareRLEnv-v0_5898a1aa_2023-01-10_10-10-48wg04npbq 2 --gen_emission --num_rollouts 10 --render_mode no_render
```
To evaluate traditional models:

# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)