# Human-Interpret

This is a Python library associated with the "Automatic discovery and description of human planning strategies" (Skirzynski, Jain, & Lieder, 2021) manuscript submitted to the Behavior Research Methods. It is is avaiable at https://arxiv.org/abs/2109.14493. Human-Interpret is an algorithm presented in this paper which transforms data from a process-tracing experiment into a set of descriptions for planning strategies used in this experiment.

We used Python 3.6.9 and Ubuntu 20.04.

## Installation

To install the library, simply clone the repository and use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages.

```bash
git clone https://github.com/RationalityEnhancement/InterpretableHumanPlanning.git
cd python
pip install -r requirements.txt
```
## Data
Download the demonstrations and other data important for the working of the algorithm as the [demos](https://owncloud.tuebingen.mpg.de/index.php/s/eq7kw5qeXX4TqcW) folder. Extract the folder to the ```python``` directory.

## Example Usage

To run the whole pipeline that includes model selection, and find descriptions of 1,2,...,K strategies used in the experiment whose data resides in the ../data folder, for instance experiment v1.0 (see the previous README for more information), type:
```bash
python3 pipeline.py --experiment v1.0 --max_num_strategies K --num_participants 0 --num_demos 128 --expert_reward 39.97
```
For a description of available parameters, see the pipeline.py file. Choosing K = 20 enables to generate the same set of results as in the paper. Model selection select which number of clusters is optimal according to some measure.

To generate descriptions of N clusters without performing model selection run the following command:
```bash
python3 pipeline.py --experiment v1.0 --max_num_strategies N --begin N --num_participants 0 --num_demos 128 --expert_reward 39.97
```

To run the code effectively first generate descriptions of ```cpp for\{ i=0; i<N+1; ++i}``` clusters, and then perform model selection, i.e.
```bash
python3 pipeline.py --experiment v1.0 --max_num_strategies i --begin i --num_participants 0 --num_demos 128 --expert_reward 39.97
python3 pipeline.py --experiment v1.0 --max_num_strategies N --begin 1 --num_participants 0 --num_demos 128 --expert_reward 39.97
```

## Disclaimer
With the additional data downloaded in the previous steps, expect the model selection to run for around 24-32 hours (procedural formula evaluation is the most time-consuming operation). To run the complete model selection without the data, expect around 40 days. When running the code with the downloaded folder and creating descriptions for N clusters only, the code should run for around 1-3 hours.
