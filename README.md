# DissentSelfCensorship

This repository contains all code supplements and Mathematica derivations for the manuscript "Strategic Analysis of Dissent and Self-Censorship" developed by Joshua J. Daymude, Robert Axelrod, and Stephanie Forrest.
Instructions for reproducing our results and experimenting with our simulations are below.


## Installation

We use [`uv`](https://docs.astral.sh/uv/) to manage Python environments.
After cloning this repository or downloading the [latest release](https://github.com/DaymudeLab/DissentSelfCensorship/releases), [install `uv`](https://docs.astral.sh/uv/getting-started/installation/) and then run the following to get all dependencies:

```shell
uv sync
```


## Reproducibility

To reproduce our data and figures, navigate to the `code/` directory and activate the Python virtual environment:

```shell
cd code
source .venv/bin/activate
```

Figs. 1 and 2 illustrate an individual's optimal action as a function of their desired dissent.
They are produced with

```shell
python opt_action.py
```

The corresponding analytical derivations (Eqs. 5&ndash;8) are verified in the Mathematica notebook `opt_action.nb`.

Fig. 3 visualizes the individual's optimal action as a function of the authority's parameters, differentiating between compliance, (partial or full) self-censorship, and defiance as different phases.
It is produced with

```shell
python phase_diagram.py
```

Figs. 4&ndash;5 and S1&ndash;3 all show different results from the adaptive authority simulations.
They are reproduced with

```shell
python hillclimbing.py -N 100000 -R 5000 -D 0.25 -B 2 -P uniform --seed 458734673
python hillclimbing.py -N 100000 -R 5000 -D 0.25 -B 8 -P uniform --seed 458734673
python hillclimbing.py -N 100000 -R 5000 -D 0.25 -B 1 -P proportional --seed 458734673
python hillclimbing.py -N 100000 -R 5000 -D 0.25 -B 3 -P proportional --seed 458734673
python hillclimbing.py --sweep -N 100000 -R 10000 -P uniform -A 1.0 -E 0.05 --seed 1978572166 --granularity 50 --trials 50 --threads 32
python hillclimbing.py --sweep -N 100000 -R 10000 -P proportional -A 1.0 -E 0.05 --seed 1978572166 --granularity 50 --trials 50 --threads 32
```

> [!WARNING]
> The sweep experiments will take a while to run and may use a lot of memory.
> We ran on a a Linux machine with a 5.7 GHz Ryzen 9 7950X CPU (16 cores, 32 threads) and 64 GB of memory, allowing us to parallelize with `--threads 32`.

If you want to experiment with the adaptive authority simulations yourself (e.g., with different parameters), the following will print usage information:

```shell
python hillclimbing.py --help
```


## Contributing

If you'd like to leave feedback, feel free to open a [new issue](https://github.com/DaymudeLab/DissentSelfCensorship/issues/new/).
If you'd like to contribute, please submit your code via a [pull request](https://github.com/DaymudeLab/DissentSelfCensorship/pulls).
