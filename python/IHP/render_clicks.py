import os
import sys
import pickle
import numpy as np
import pandas as pd
from graphviz import Digraph
from collections import defaultdict
from IHP.experiment_utils import Experiment
from IHP.modified_mouselab import TrialSequence
import imageio
import shutil

""" This file contains functions to plot/make a video of a
    click sequence in a given environment
"""

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def render_nodes(trial, node_attrs = {}):
    def color(val):
        if val > 0:
            return '#8EBF87'
        else:
            return '#F7BDC4'

    dot = Digraph(format = 'png')
    for node_label,node in trial.node_map.items():
        observed = node.observed
        value = node.value
        attrs = {
                'color': color(value) if observed else 'grey',
                'label': str(round(value,2)) if observed else str(node.label),
                'style': 'filled'
        }
        attrs.update(node_attrs.get(node_label,{}))
        dot.node(str(node.label), **attrs)
    for node_label,node in trial.node_map.items():
        for child in node.children:
            dot.edge(str(node.label),str(child.label))
    return dot

def render_click_sequence(trial, path, render_path):
    trial.reset_observations()
    node_num = 0
    print(render_path)
    for node in path:
        trial.node_map[node].observe()
        render_nodes(trial).render(f"{render_path}/{node_num}", cleanup = True)
        node_num += 1
    trial.reset_observations()

def render_sequence_clicks(trial_sequence, click_sequences, path, gif=True):
    for i, (trial, clicks)  in enumerate(zip(trial_sequence.trial_sequence, click_sequences)):
        d_path = f"{path}/{i}"
        create_dir(d_path)
        render_click_sequence(trial, clicks, d_path)
        if gif:
            images = []
            num_files = len(os.listdir(d_path))
            for click_num in range(num_files):
                file_path = os.path.join(d_path, f"{click_num}.png")
                images.append(imageio.imread(file_path))
            imageio.mimsave(f"{path}/{i}.gif", images, fps=2)
            shutil.rmtree(d_path)

def render_experiment_clicks(exp_num, pipeline, path, block=None):
    if block:
        E = Experiment(exp_num, block=block)
    else:
        E = Experiment(exp_num)
    
    envs = E.planning_data['envs']
    clicks = E.planning_data['clicks']

    for pid in envs.keys():
        print(pid)
        trial_sequence = TrialSequence(len(envs[pid]), pipeline, ground_truth = envs[pid])
        render_sequence_clicks(trial_sequence, clicks[pid], path=f"{path}/{exp_num}/{pid}")

def render_cluster_sequences(cluster_info, pipeline, path):
    for k in cluster_info.keys():
        envs = []
        clicks = []
        for i in range(len(cluster_info[k])):
            envs.append(cluster_info[k][i][0])
            clicks.append(cluster_info[k][i][1])
        trial_sequence = TrialSequence(len(envs), pipeline, ground_truth=envs)
        render_sequence_clicks(trial_sequence, clicks, f"{path}/{k}")
