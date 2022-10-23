import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IHP.learning_utils import pickle_load, pickle_save, \
                               get_normalized_features, create_dir
from IHP.sequence_utils import ClickSequence, compute_trial_features, \
                               compute_total_error_gradient, compute_error_gradient
from IHP.mixture_models import SoftmaxMixture, MixtureEM
from IHP.experiment_utils import Experiment

def get_total_data(pipeline, envs, click_sequences, features, normalized_features):
    """ 
    This function is used to convert each (click sequence, env) pair into a click 
    sequence object which can then be used to easily extract features of 
    belief-state computation pairs
    
    Parameters
    ----------
    pipeline : [ ( [ int ], function ) ]
        Reward structure specification
    envs : [ [ int ] ]
        Rewards underlying the nodes
    click_sequences : [ [ int ] ]
        Click sequences in each environment
    features : [ str ]
        Features in the feature space
    normalized_features : dict
        Max and min values of each feature
    
    Returns
    -------
    clicks : [ ClickSequence ]
        List of ClickSequence objects
    """
    clicks = []
    for env, click_sequence in zip(envs, click_sequences):
        clicks.append(ClickSequence(click_sequence, env, pipeline, features, normalized_features))
    return clicks

def initialize_experiment(exp_num, block):
    """
    Create an Experiment object.
    
    Parameters
    ----------
    exp_num : str
        Identifier of the experiment with the human data
    block : str
        Part of the experiment to extract the data from (e.g. 'train')
    
    Returns
    -------
    E : IHP.experiment_utils.Experiment
        Instance of the Experiment object
    """
    additional_constraints = {}
    # Initializing the experiment
    if block:
        E = Experiment(exp_num, block = block)
    else:
        if "c2.1" in exp_num:
            if exp_num == "c2.1_inc":
                additional_constraints['variance'] = 2424
            else:
                additional_constraints['variance'] = 2442
            E = Experiment("c2.1", **additional_constraints)
        else:
            E = Experiment(exp_num)
    return E

def initialize_participant_data(envs, actions, num_participants):
    """
    Create 3 lists with environments participants interacted with, their actions
    and their experiment's identifiers alongside numer of the trial identifiers.
    """
    # Getting pids of participants in the experiment
    total_pids = sorted(list(envs.keys()))
    if num_participants == 0:
        num_participants = len(total_pids)

    p_envs = []
    p_actions = []
    p_points = []
    for pid in total_pids[:num_participants]:
        p_envs += envs[pid]
        p_actions += actions[pid]
        p_points += [(pid, i) for i in range(len(envs[pid]))]
    return p_envs, p_actions, p_points, num_participants

def load_em_features(exp_num):
    """
    Load features used for defining the softmax polcies of the EM clusters.
    
    Parameters
    ----------
    exp_num : str
        Identifier of the experiment with the human data
    
    Returns
    -------
    features : [ str ]
        Names of the features defined in the Mouselab MDP environment used
        in the models of the EM clusters
    normalized_features : dict
        str : float
            Value for which the feature needs to be divided by to get a value in
            [0,1]
    pipeline : [ ( [ int ], function ) ]
            Reward structure specification
    """
    # Features which will be used for clustering
    features = pickle_load("data/em_features.pkl")
    
    # Pipeline is used to specify the reward structure of the environment
    exp_reward_structures = pickle_load("data/exp_reward_structures.pkl")
    exp_pipelines = pickle_load("data/exp_pipelines.pkl")
    if "c2.1" not in exp_num:
        reward_structure = exp_reward_structures[exp_num]
        normalized_features = get_normalized_features(reward_structure)
        pipeline = [exp_pipelines[exp_num][0]]*1000
    else:
        if exp_num == "c2.1_inc":
            pipeline = [exp_pipelines["c2.1_inc"][0]]*100
            reward_structure = 'high_increasing'
        else:
            pipeline = [exp_pipelines["c2.1_dec"][0]]*100
            reward_structure = 'high_decreasing'
        normalized_features = get_normalized_features(reward_structure)
    return features, normalized_features, pipeline
    
def get_num_participants(exp_id, num_clusters, block):
    """
    Return the number of participants taken for analyzing in search of planning
    strategies. (In particular, if all were taken, return how many participants
    can be found in experiment's data).
    
    Parameters
    ----------
    exp_id : str
        Identifier of the experiment with the human data
    num_clusters : int
        Number of found EM clusters
    block : str
        Part of the experiment to extract the data from (e.g. 'train')
    
    Returns
    -------
    num_p : int
        Number of participants
    """
    E = initialize_experiment(exp_id, block)

    # Getting planning data
    envs = E.planning_data['envs']
    clicks = E.planning_data['clicks']

    _, _, _, num_p = initialize_participant_data(envs, clicks, 0)
    return num_p

def compute_clusters(exp_id, num_participants, num_clusters, block):
    """
    Perform EM clustering for num_clusters on the data for num_participants
    in block block of experiment exp_id.
    
    Represent cluster centers as weights associated with the features computed
    on the states. Use a softmax mixture model.
    
    Parameters
    ----------
    exp_id : str
        Identifier of the experiment with the human data
    num_participants : int
        Number of participants whose data to extract
    num_clusters : int
        Number of EM clusters to define
    block : str
        Part of the experiment to extract the data from (e.g. 'train')
    
    Returns
    -------
    optimized_model : IHP.mixture_models.SoftmaxMixture
        Updated SMM model with the learned weights, mixture model weihgts and 
        posterior probabilities
    labels : [ [ int ] ]
        Cluster assignment for each action sequence for each participant
    true_ll : float
        Likelihood of the data assigned to the clusters; total likelihood
    p_envs : [ [ int ] ]
        List of environments participats interacted with
    p_clicks : [ [ int ] ]
        List of actions participants taken in the environments
    p_points : [ [ (int, int) ] ] 
        List that IDs click sequences; contains pairs of (participant_id, num_trial)
    SMM.get_params()
        weights : np.array()
            weights.shape = (num_clusters, num_features)
            Softmax weights for the features for each cluster
        pis : np.array()
            pis.shape = (1, num_clusters)
            Softmax mixture model weights
        posteriors : np.array()
            Posterior probabilities for all the action sequences for belonging
            to each of the num_clusters clusters
    num_p : int
        Actual number of participats whose data was considered (if the parameter
        passed in the function denoted "all")
    """
    E = initialize_experiment(exp_id, block)

    # Getting planning data
    envs = E.planning_data['envs']
    clicks = E.planning_data['clicks']

    res = initialize_participant_data(envs, clicks, num_participants)
    p_envs, p_clicks, p_points, num_p = res
    features, normalized_features, pipeline = load_em_features(exp_id)

    # Representing data to pass to the EM algorithm
    total_data = get_total_data(pipeline, p_envs, p_clicks, 
                                features, normalized_features)
    SMM = SoftmaxMixture(num_clusters, len(features), global_op = False)
    EM = MixtureEM(SMM, niters=None)
    optimized_model, labels, true_ll = EM.optimize(total_data)
    return optimized_model, labels, true_ll, p_envs, p_clicks, p_points, \
           SMM.get_params(), num_p

def save_clusters_data(exp_id, num_cl, num_p, labels, log_lik, 
                       participant_data, smm_params, info):
    """
    Save parameters of the computed EM clusters.
    
    Parameters
    ----------
    (See compute_clusters)
    exp_id : str
    num_clusters : int
    num_p : int
    labels : [ [ int ] ]
    log_lik : float
    participant_data : tuple
    smm_params : tuple
    """

    p_envs, p_clicks, p_points = participant_data
    label_points = defaultdict(list)
    for E, C, p, l in zip(p_envs, p_clicks, p_points, labels):
        label_points[l].append([E, C, p])

    d = f"clustering/em_clustering_results/{exp_id}"
    create_dir(d)

    # Saving cluster weights and posterior probabilities
    pickle_save(smm_params, f"{d}/{num_cl}_{num_p}_params" + info + ".pkl")

    # Saving cluster assignments, Evidence lower bound and the posterior probabilities
    pickle_save((label_points, log_lik), f"{d}/{num_cl}_{num_p}" + info + ".pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', '-e',
                        type=str,
                        help="Identifier of the experiment to interpret.")
    parser.add_argument('--num_clusters', '-c',
                        type=int,
                        help="Number of clusters to divide the data into.")
    parser.add_argument('--num_participants', '-p',
                        type=int,
                        help="Number of participants whose data to take into account.",
                        default=0)
    parser.add_argument('--block', '-b',
                        default=None)

    # if num_participants is 0, the clustering is done on data from all the participants
    res = compute_clusters(args.experiment_id,
                           args.num_participants,
                           args.num_clusters,
                           args.block)
    opt_model, labels, ll, pd_env, pd_cl, pd_pt, par, num_p = res
    save_clusters_data(exp_idf=args.experiment_id, num_cl=args.num_clusters, 
                       num_p=num_p, labels=labels, log_lik=ll, 
                       participant_data=[pd_env, pd_cl, pd_pt], smm_params=par)

    
