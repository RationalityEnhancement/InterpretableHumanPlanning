# Interpretable human planning

This is a repository associated with a project on automatically discovering and describing planning strategies used by people. Our computational method for enabling this process is described in a manuscript submitted to Behavior Research Methods journal (Skirzynski, Jain, & Lieder, 2022) and available at https://arxiv.org/abs/2109.14493. 

## Human-Interpret

Navigate to python folder to learn more about Human-Interpret, a method for that takes data from a process-tracing experiment and returns descriptions of planning strategies that people used in this experiment. Human-Interpret finds strategies by performing probabilistic clustering on the process-tracing data, transforming those clusters into procedural instructions with AI-Interpret (imitation learning method; Skirzynski, Becker, & Lieder, 2021) and DNF2LTL (obtaining procedural descriptions from flowcharts; Becker, Skirzynski, van Opheusden, & Lieder, 2021), and applying Bayesian model selection on multiple iterations of this procedure.

## Experiments

Navigate to the data folder in order to see the data gathered in process-tracing experiments that externalized human planning. There are 5 available experiments:

1. "v1.0" - 3-step task with increasing variance rewards (standard Mouselab task); expert reward: 39.97; participants: 180
2. "F1" - Same as "v1.0"
3. "c1.1" - 3-step constant variance task (rewards in [-10, -5, 5, 10]); expert reward: 9.33; participants: 62
4. "T1.1" - 5-step transfer task. Only the test block of this experiment is to be used.; expert reward: 50; participants: 120
5. "c2.1_dec" - 3-step task with decreasing variance rewards; expert reward: 30.14; participants: 35
