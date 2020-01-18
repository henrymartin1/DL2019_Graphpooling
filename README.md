[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/henrymartin1/DL2019_Graphpooling/master)

# Deep learning 2019 group work on graph pooling methods

## Installation
You can create the environment using conda: `conda env create -f environment.yml`

However, we use [pytorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/) and it can be difficult to install. We would recommend that you create an environment without pytorch-geometric and then install it following the official instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Run the code
The core of our project are three experiments.
- __complexity:__ Here we tested the memory requirements, the time complexity and sample efficiency of the different pooling methods.
- __datasets:__ Here we tested the performance of the different pooling methods on different datasets.
- __varying_training_percentage:__ Here we tested the performance of the different datasets on Cora as a function of the training label percentage.

Every experiment has its proper folder in the `experiments` folder (e.g., `experiments/datasets`). Every experiment contains a `config` file and a `main` file. In order to reproduce the experiments it is enough to just run the mainfile. However, due to personal preference of the authors the experiments have different working folders.

Here are the files by experiment:
- complexity
	- `config:` complexity_config.py
	- `main:` complexity_run.py
	- `wdir:` `.../DL2019_Graphpooling`
	- `plots:` complexity_plots.py 
- datasets:
	- `config:` datasets_config.py
	- `main:` datasets_run.py.py
	- `wdir:` `.../DL2019_Graphpooling`
- varying_training_percentage
	- `config:` varying_training_percentage_config.py
	- `main:` varying_training_percentage_experiments.py
	- `wdir:` `.../DL2019_Graphpooling/experiments/`
	- `plots:` integrated in main file
	varying_training_percentage

Feel free to play around with the config file to generate alternative experiments. 

## Other

For GPU support replace `cpuonly` with `cudatoolkit=10.1` using your cuda version. You can run a motivating example using [binder](https://mybinder.org/v2/gh/henrymartin1/DL2019_Graphpooling/master)


