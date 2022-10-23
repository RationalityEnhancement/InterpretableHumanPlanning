import random
import os
import argparse
import numpy as np
import pickle
import copy
import scipy.integrate
from RL2DT.strategy_demonstrations import make_modified_env as make_env
from interpret_formula import load_EM_data
from scipy.special import softmax
from IHP.modified_mouselab import TrialSequence
from RL2DT.PLP.DSL import *
from progress.bar import ChargingBar
from scipy.stats import norm as Normal
from scipy.special import beta as Beta
from hyperparams import ITERS

ALPHA = 1
BETA = 7

class ProceduralStrategy(object):
    """
    Class handling a procedural description with elements particular to the linear
    temporal logic formulas.
    """
    def __init__(self, text_formula):
        self.strategies = text_formula.split(' AND NEXT ')
        self.current_index = 0
        self.strategies = self.strategies[:-1] + self.strategies[-1].split('\n\n')
        self.current_index = 0
        st_unl = self.strategies[0].split(" UNLESS ")
        self.current = st_unl[0]
        st_unt = self.current.split(" UNTIL ")
        self.current = st_unt[0]
        self.unless = st_unl[1] if len(st_unl) > 1 else ""
        self.until = st_unt[1] if len(st_unt) > 1 else "ONE-STEP"
        self.steps = 0
        
    def next(self):
        self.current_index += 1
        loop = False
        
        if self.current_index == len(self.strategies):
            raise Exception("The end of the strategy")
        
        if "LOOP" in self.strategies[self.current_index]:
            loop = self.strategies[self.current_index][10:]
            lp_unl = loop.split(" UNLESS ")
            loop_formula = lp_unl[0]
            for i, s in enumerate(self.strategies):
                clean_s = s[2:-2] if s[:2] == "((" else s
                if loop_formula in clean_s:
                    self.current_index = i
                    break
            self.unless = lp_unl[1] if len(lp_unl) > 1 else ""
            loop = True
            
        st_unl = self.strategies[self.current_index].split(" UNLESS ")
        self.current = st_unl[0]
        st_unt = self.current.split(" UNTIL ")
        self.current = st_unt[0]
        
        if not(loop): self.unless = st_unl[1] if len(st_unl) > 1 else ""
        self.until = st_unt[1] if len(st_unt) > 1 else "ONE-STEP"
        self.steps = 0
        
    def until_condition_satisfied(self, state, unobs):
        if "ONE-STEP" in self.until:
            return self.steps != 0
        elif "STOPS APPLYING" in self.until:
            return not(any([eval('lambda st, act : ' + self.current)(state, a) 
                        for a in unobs]))
        else:
            return all([eval('lambda st, act : ' + self.until)(state, a) 
                        for a in unobs])
        
    def applies(self, state, unobs):
        return any([eval('lambda st, act : ' + self.current)(state, a) 
                        for a in unobs])
    
    def block_strategy(self, state, unobs):
        if self.unless == '':
            return False
        else:
            vals = [eval('lambda st, act : ' + self.unless)(state, a) 
                    for a in unobs]
            vals = [bool(v) for v in vals]
            return any(vals)
        
    def step(self):
        self.steps += 1


def simulate_softmax(start_state, W, softmax_features, softmax_norm_features):
    """
    Perform a rollout of the softmax policy found with the EM algorithm. 
    
    Collect the taken actions.

    Parameters
    ----------
    start_state : IHP.modified_mouselab.Trial
        Environment representing the Mouselab MDP used for generating states
    softmax_features : [ str ]
        Names of the features defined in the Mouselab MDP environment (Trial) used
        in the models of the EM clusters
    W : [ float ]
        Weights associated with the features
    softmax_norm_features : dict
        str : float
            Value for which the feature needs to be divided by to get a value in
            [0,1]
        
    Returns
    -------
    actions : [ int ]
        Actions taken by the policy in the rollout
    """
    actions = []
    action = -100
    while(action != 0):
        unobserved_nodes = start_state.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        feature_values = start_state.get_node_feature_values(unobserved_nodes, 
                                                             softmax_features,
                                                             softmax_norm_features)
        dot_product = np.dot(W, feature_values.T)
        softmax_dot = softmax(dot_product)
            
        max_val = max(softmax_dot)
        n = sum([1 if i==max_val else 0 for i in softmax_dot])
        greedy_dot = [1./n if i==max_val else 0 for i in softmax_dot]
    
        action = np.random.choice(unobserved_node_labels, p = greedy_dot)
        start_state.node_map[action].observe()
        
        actions.append(action)
    return actions
    
def simulate_formula(start_state, strategies, allowed_acts, i=False):
    """
    Perform a rollout of the policy induced by the procedural formula. 
    
    Collect the taken actions.

    Parameters
    ----------
    start_state : IHP.modified_mouselab.Trial
        Environment representing the Mouselab MDP used for generating states
    strategies : [ ProceduralStrategy ]
        Procedural descriptions that are part of the whole procedural formula
        where they are separated by ORs
    allowed_acts : [ int ]
        Actions allowed to be taken by the policy induced through the procedural
        formula
        
    Returns
    -------
    actions : [ int ]
        Actions taken by the formula-induced policy in the rollout
    """
    if strategies[0].current == "None":
        return []
    all_actions = list(start_state.node_map.keys())
    greedy_dot = [1./len(allowed_acts) if a in allowed_acts else 0
                  for a in all_actions]
    if i: print(i)
    action = np.random.choice(all_actions, p = greedy_dot)
    start_state.node_map[action].observe()
    actions = [action]
    while(action != 0):
        allowed_acts = []
        unobserved_nodes = start_state.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        applicable_strategies = [s for s in strategies 
                                 if s.applies(start_state, unobserved_node_labels)]
        if applicable_strategies == []:
            break
        
        untils = {s: s.until_condition_satisfied(start_state, unobserved_node_labels) 
                  for s in applicable_strategies}
            
        if any(list(untils.values())):
            applicable_strategies = [s for s in applicable_strategies if untils[s]]
            try:
                for s in applicable_strategies:
                    s.next()
                    if not(s.block_strategy(start_state, unobserved_node_labels)):
                        allowed_acts += return_allowed_acts(s.current, 
                                                            start_state,
                                                            unobserved_node_labels)
                    else:
                        allowed_acts = [0]
                    s.step()
            except:
                allowed_acts = [0]
        else:
            for s in applicable_strategies:
                if not(s.block_strategy(start_state, unobserved_node_labels)):
                    allowed_acts += return_allowed_acts(s.current, 
                                                        start_state,
                                                        unobserved_node_labels)
                else:
                    allowed_acts = [0]
                s.step()
        allowed_acts = list(set(allowed_acts))
        greedy_dot = [1/float(len(allowed_acts)) if a in allowed_acts else 0
                      for a in all_actions]
                      
        action = np.random.choice(all_actions, p = greedy_dot)
        start_state.node_map[action].observe()
        
        actions.append(action)
    return actions
    

def compute_score_formula_rollouts(formula, W, pipeline, softmax_features, 
                                   softmax_norm_features, verbose):
    """
    Measure the likelihood of the procedural formula rollouts under the softmax
    policy.
    
    Compute the proportion of optimal actions.

    Parameters
    ----------
    formula : str
        Procedural formula
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes. For each rollout.
    (See simulate_softmax)
    W : [ float ]
    softmax_features : [ str ]
    softmax_norm_features : dict
        str : float
        
    Returns
    -------
    log_likelihood_softmax : float
        Log likelihood for the formula rollouts under the softmax policy
    log_likelihood_greedy : float
        Log likelihood for the formula rollouts under the policy derived from
        the softmax policy with uniform distribution over max probability actions
    optimal_score : float
        Proportion of actions taken according to the max probability of the 
        softmax policy
    mean_len : float
        Mean length of the formula rollouts
    """
    log_likelihood_softmax = 0
    log_likelihood_greedy = 0
    optimal_acts, bad_acts = 0, 0
    total_len = 0
    if verbose: bar = ChargingBar('Computing formula rollouts', max=ITERS)
    for i in range(ITERS):
        rollout = []
        env = TrialSequence(1, pipeline).trial_sequence[0]
        
        new_iter = True
        formulas = formula.split('\n\nOR\n\n')
        strategies = []
        for f in formulas:
            strategies.append(ProceduralStrategy(f))
            
        action = -100
        while action != 0:
            total_len += 1
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
            all_nodes = unobserved_nodes + env.get_observed_nodes()
            all_nodes_labels = [node.label for node in all_nodes]
            feature_values = env.get_node_feature_values(unobserved_nodes, 
                                                         softmax_features,
                                                         softmax_norm_features)
            dot_product = np.dot(W, feature_values.T)
            softmax_dot = softmax(dot_product)
            
            max_val = max(softmax_dot)
            n = sum([1 if i==max_val else 0 for i in softmax_dot])
            greedy_dot = [1./n if i==max_val else 1e-6 for i in softmax_dot]
            
            softmax_dot = {n: v for n,v in zip(unobserved_node_labels, softmax_dot)}
            greedy_dot = {n: v for n,v in zip(unobserved_node_labels, greedy_dot)}
            
            if new_iter:
                applicable_strategies = [s for s in strategies 
                                          if s.applies(env, unobserved_node_labels)]
                if applicable_strategies == []:
                    #No planning strategy
                    strategy = strategies[0]
                else:
                    strategy = random.choice(applicable_strategies)
            untils = {s: s.until_condition_satisfied(env, unobserved_node_labels) 
                      for s in applicable_strategies}
            
            if any(list(untils.values())):
                applicable_strategies = [s for s in applicable_strategies if untils[s]]
                try:
                    for s in applicable_strategies: s.next()
                    strategy = random.choice(applicable_strategies)
                    if not(strategy.block_strategy(env, unobserved_node_labels)):
                        action = take_action(strategy.current, env, 
                                             unobserved_node_labels)
                    else:
                        action = 0
                    strategy.step()
                except:
                    action = 0
            else:
                if not(strategy.block_strategy(env, unobserved_node_labels)):
                    action = take_action(strategy.current, env, 
                                         unobserved_node_labels)
                else:
                    action = 0
                strategy.step()
            log_likelihood_softmax += np.log(softmax_dot[action])
            log_likelihood_greedy += np.log(greedy_dot[action])
            if greedy_dot[action] != 1e-6:
                optimal_acts += 1
            elif action == 0 and formula != "None":
                ##early termination
                rollouts = [simulate_softmax(copy.deepcopy(env), W, softmax_features, 
                                             softmax_norm_features) for _ in range(10)]
                add_clicks = np.mean([len(r) for r in rollouts])
                bad_acts += add_clicks
            else:
                bad_acts += 1
            env.node_map[action].observe()
            new_iter = False
            rollout.append(action)
        if verbose: bar.next()
    if verbose: bar.finish()
    
    optimal_score = optimal_acts/(optimal_acts+bad_acts)
    mean_len = total_len/ITERS
    return log_likelihood_softmax, log_likelihood_greedy, optimal_score, mean_len


def compute_score_softmax_rollouts(formula, W, pipeline, softmax_features, 
                                   softmax_norm_features, verbose):
    """
    Measure the likelihood of the softmax policy rollouts under the policy induced
    by the procedural formula.
    
    Compute the proportion of optimal actions.

    Parameters
    ----------
    formula : str
        Procedural formula
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes. For each rollout.
    (See simulate_softmax)
    W : [ float ]
    softmax_features : [ str ]
    softmax_norm_features : dict
        str : float
        
    Returns
    -------
    log_likelihood : float
        Log likelihood for the softmax policy rollouts under the policy induced
        by the procedural formula
    optimal_score : float
        Proportion of actions taken according to the allowed actions of the  
        formula-induced policy
    mean_len : float
        Mean length of the softmax policy rollouts
    """
    optimal_acts, bad_acts = 0, 0
    total_len = 0
    log_likelihood = 0
    if verbose: bar = ChargingBar('Computing softmax rollouts', max=ITERS)
    for i in range(ITERS):
        env = TrialSequence(1, pipeline).trial_sequence[0]
        
        new_iter = True
        formulas = formula.split('\n\nOR\n\n')
        strategies = []
        for f in formulas:
            strategies.append(ProceduralStrategy(f))
            
        action = -100
        while action != 0:
            total_len += 1
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
            feature_values = env.get_node_feature_values(unobserved_nodes, 
                                                         softmax_features,
                                                         softmax_norm_features)
            dot_product = np.dot(W, feature_values.T)
            softmax_dot = softmax(dot_product)
            
            max_val = max(softmax_dot)
            n = sum([1 if i==max_val else 0 for i in softmax_dot])
            greedy_dot = [1./n if i==max_val else 0 for i in softmax_dot]
            
            action = np.random.choice(unobserved_node_labels, p = greedy_dot)
            
            if new_iter:
                applicable_strategies = [s for s in strategies 
                                         if s.applies(env, unobserved_node_labels)]
            if applicable_strategies == []:
                #No planning strategy
                allowed_acts = [0]
            else:
                allowed_acts = []
            untils = {s: s.until_condition_satisfied(env, unobserved_node_labels) 
                      for s in applicable_strategies}
            
            if any(list(untils.values())):
                applicable_strategies = [s for s in applicable_strategies 
                                         if untils[s]]
                try:
                    for s in applicable_strategies:
                        s.next()
                        if not(s.block_strategy(env, unobserved_node_labels)):
                            allowed_acts += return_allowed_acts(s.current, 
                                                                env,
                                                                unobserved_node_labels)
                        else:
                            allowed_acts = [0]
                        s.step()
                except:
                    allowed_acts = [0]
            else:
                for s in applicable_strategies:
                    if not(s.block_strategy(env, unobserved_node_labels)):
                        allowed_acts += return_allowed_acts(s.current, 
                                                            env,
                                                            unobserved_node_labels)
                    else:
                        allowed_acts = [0]
                    s.step()
            
            if formula == "None": allowed_acts = []
            allowed_acts = list(set(allowed_acts))
            distr = {a: 1./len(allowed_acts) if a in allowed_acts else 1e-6
                     for a in env.node_map.keys()}
            if action in allowed_acts:
                optimal_acts += 1
            elif action == 0 and formula != "None":
                ##early termination
                rollouts = [simulate_formula(copy.deepcopy(env), 
                                             copy.deepcopy(applicable_strategies), 
                                             allowed_acts) for _ in range(10)]
                add_clicks = np.mean([len(r) for r in rollouts])
                bad_acts += add_clicks
            else:
                bad_acts += 1
            log_likelihood += np.log(distr[action])
            env.node_map[action].observe()
            new_iter = False
        if verbose: bar.next()
    if verbose: bar.finish()
    
    optimal_score = optimal_acts/(optimal_acts+bad_acts)
    mean_len = total_len/ITERS
    return log_likelihood, optimal_score, mean_len


def take_action(strategy_str, state, unobserved_nodes):
    """
    Output an action congruent with a logical formula.

    Parameters
    ----------
    strategy_str : str
        String representing the logical formula
    state : IHP.modified_mouselab.Trial
        Environment representing the Mouselab MDP
    unobserved_nodes : [ int ]
        Identifiers of the Mouselab MDP nodes which were not yet clicked
        
    Returns
    -------
    random.choice(allowed) : int
        Action allowed by the input formula strategy_str
    """
    actions = {a: eval('lambda st, act : ' + strategy_str)(state, a) 
               for a in unobserved_nodes}
    allowed = [a for a in actions.keys() if actions[a] == True]
    if allowed == []:
        return 0
    return random.choice(allowed)

def return_allowed_acts(strategy_str, state, unobserved_nodes):
    """
    Output an action congruent with a logical formula.

    Parameters
    ----------
    (See take_action)
    strategy_str : str
    state : IHP.modified_mouselab.Trial
    unobserved_nodes : [ int ]
        
    Returns
    -------
    allowed : [ int ]
        List of actions allowed by the input formula strategy_str
    """
    actions = {a: eval('lambda st, act : ' + strategy_str)(state, a) 
               for a in unobserved_nodes}
    allowed = [a for a in actions.keys() if actions[a] == True]
    if allowed == []:
        return [0]
    return allowed

def load_participant_data(exp_id, num_clust, num_part, clust_id, info, freqs=False):
    """
    Extract human data belonging to a particular cluster found with the EM 
    algorithm.

    Parameters
    ----------
    exp_id : str
        Identifier of the experiment from which the human data (used to create 
        the EM softmax policies) came from
    num_clust : int
        Identifier of the model that is the total number of EM clusters
    num_part : int
        Number of participants of exp_id whose data was considered
    clust_id : int
        Identifier of the EM cluster (softmax policy)
    freqs (optional) : bool
        Whether to return strategy frequencies
        
    Returns
    -------
    envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    action_seqs : [ [ int ] ]
        List of action sequences taken by the participants from cluster clust_id
        in each consecutive environment
    """
    with open('./clustering/em_clustering_results/' + exp_id + '/' + \
              str(num_clust) + '_' + str(num_part) + info + '.pkl', 'rb') as handle:
        dict_object = pickle.load(handle)
    dct = dict_object[0]
    envs = [data[0] for data in dct[clust_id]]
    action_seqs = [data[1] for data in dct[clust_id]]
    part_trial = [data[-1] for data in dct[clust_id]]
    
    lens = {k : len(v) for k,v, in dct.items()}
    total_len = sum(lens.values())
    freq = {l: round(float(v)/total_len, 4) for l, v in lens.items()}
    
    if freqs:
        return envs, action_seqs, part_trial, freq[clust_id]
    return envs, action_seqs, part_trial

from functools import lru_cache
@lru_cache()
def compute_score_people(envs, pipeline, people_acts, formula, verbose):
    """
    Measure the likelihood of the human data inside a cluster under the policy 
    induced by the procedural formula.
    
    Compute the proportion of optimal actions.
    
    A model is defined as a procedural description of the softmax policy and the
    probability function is given as 
    Epsilon * Uniform(non-allowed-actions(description)
    + (1-Epsilon) * Uniform(allowed_actions(description)).
    
    Parameters
    ----------
    envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes. For each rollout.
    people_acts : [ [ int ] ]
        List of action sequences taken by people in each consecutive environment
    formula : str
        Procedural formula
    verbose : bool
        Whether to print the progress bar or not
        
    Returns
    -------
    log_likelihood : float
        Log likelihood for human data in the clsuter under the policy induced by 
        the procedural formula
    opt_score : float
        Proportion of actions taken according to the allowed actions of the  
        formula-induced policy
    mean_len : float
        Mean length of human demonstrations
    Epsilon : float
        Measure of fit of the procedural description with respect to human data;
        free parameter for the action model
    opt_act_score : float
        Average likelihood per planning operation over average optimal likelihood
        per planning operation when optimal is computed as though all (state, action)
        pairs agreed with the formula 
    mean_lik : float
        Mean of the likelihoods for human planning operations under the formula
    geo_mean_lik : float
        Geomatric mean of the likelihoods for human planning operations under the formula
    """
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    optimal_acts = 0
    total_len = 0
    log_likelihood = 0
    corr_likelihood, incorr_likelihood, optimal_likelihood = 0, 0, 0
    ll_allowed, ll_not_allowed = 0, 0
    epsilon_power, _1_epsilon_power = 0, 0
    if verbose: bar = ChargingBar('Computing people data', max=len(envs))
    for ground_truth, actions in zip(envs, people_acts):
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        new_iter = True
        formulas = formula.split('\n\nOR\n\n')
        strategies = []
        for f in formulas:
            strategies.append(ProceduralStrategy(f))
            
        for action in actions:
            total_len += 1
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
                        
            if new_iter:
                applicable_strategies = [s for s in strategies 
                                         if s.applies(env, unobserved_node_labels)]
            if applicable_strategies == []:
                #No planning strategy
                allowed_acts = [0]
            else:
                allowed_acts = []
            untils = {s: s.until_condition_satisfied(env, unobserved_node_labels) 
                      for s in applicable_strategies}
            
            if any(list(untils.values())):
                applicable_strategies = [s for s in applicable_strategies 
                                         if untils[s]]
                try:
                    for s in applicable_strategies:
                        s.next()
                        if not(s.block_strategy(env, unobserved_node_labels)):
                             allowed_acts += return_allowed_acts(s.current, 
                                                                 env,
                                                                 unobserved_node_labels)
                        else:
                            allowed_acts = [0]
                        s.step()
                except:
                    allowed_acts = [0]
            else:
                for s in applicable_strategies:
                    if not(s.block_strategy(env, unobserved_node_labels)):
                        allowed_acts += return_allowed_acts(s.current, 
                                                            env,
                                                            unobserved_node_labels)
                    else:
                        allowed_acts = [0]
                    s.step()
            
            if formula == "None": allowed_acts = []
            allowed_acts = list(set(allowed_acts))
            allowed_distr = {a: 1/float(len(allowed_acts)) for a in allowed_acts}
            not_allowed_acts = [a for a in env.node_map.keys()
                                if a not in allowed_acts]
            not_allowed_distr = {a: 1/float(len(not_allowed_acts)) 
                                 for a in not_allowed_acts}
            if action in allowed_acts:
                optimal_acts += 1
                ll_allowed += np.log(allowed_distr[action])
                corr_likelihood += allowed_distr[action]
                _1_epsilon_power += 1
            else:
                ll_not_allowed += np.log(not_allowed_distr[action])
                incorr_likelihood += not_allowed_distr[action]
                epsilon_power += 1
            
            optimal_likelihood += allowed_distr[allowed_acts[0]]
            env.node_map[action].observe()
            new_iter = False
        if verbose: bar.next()
    if verbose: bar.finish()
    
    opt_score = optimal_acts / total_len
    mean_len = total_len / len(envs)
    if verbose: print("Epsilon optimization...")
    ## computing epsilon maximizing likelihood * prior assuming prior is a 
    ## beta funciton, to later compute the marginal likelihood 
    nom = float(epsilon_power + ALPHA - 1)
    den = (epsilon_power + _1_epsilon_power + ALPHA + BETA - 2)
    ## Analytical solution to the global maximum in [0,1]
    epsilon = nom/den
    if epsilon_power == 0:
        log_likelihood = ll_allowed
    elif _1_epsilon_power == 0:
        log_likelihood = ll_not_allowed
    else:
        log_likelihood = epsilon_power*np.log(epsilon) + ll_not_allowed + \
                         _1_epsilon_power*np.log((1-epsilon)) + ll_allowed
    
    total_likelihood = (1-epsilon)*corr_likelihood + epsilon*incorr_likelihood                   
    mean_lik = total_likelihood / total_len
    mean_opt_lik = optimal_likelihood / total_len
    opt_act_score = mean_lik / mean_opt_lik
    geo_mean_lik = np.exp(log_likelihood / total_len)
                         
    return log_likelihood, opt_score, mean_len, epsilon, opt_act_score, mean_lik, geo_mean_lik


def compute_likelihood_random(envs, pipeline, people_acts):
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    optimal_acts = 0
    total_len = 0
    likelihood, log_likelihood = 0, 0
    bar = ChargingBar('Computing people data', max=len(envs))
    for ground_truth, actions in zip(envs, people_acts):
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        likelihood_seq, ll_seq = 0, 0
        
        for act in actions:
            total_len += 1
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
            random_distr = softmax([1./len(unobserved_nodes) for _ in range(len(unobserved_nodes))])
            ind = unobserved_node_labels.index(act)
            likelihood_seq += random_distr[ind]
            ll_seq += np.log(random_distr[ind])
            
            env.node_map[act].observe()
            
        likelihood += likelihood_seq
        log_likelihood += ll_seq
        bar.next()
    bar.finish()
    mean_lik = likelihood / total_len
    geo_mean_lik = np.exp(log_likelihood / total_len)
    return mean_lik, geo_mean_lik


def compute_likelihood_microscope(envs, pipeline, people_acts, weights, softmax_feats, softmax_norm_feats, ids, assignment):
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    with open('data/test_temperatures.pkl', 'rb') as f:
        test_temps = pickle.load(f)
    with open('data/train_temperatures.pkl', 'rb') as f:
        train_temps = pickle.load(f)
    
    def softmax2(x, temp):
        e_x = np.exp((x - np.max(x))/ temp)
        return e_x / e_x.sum()
    
    optimal_acts = 0
    total_len = 0
    likelihood, log_likelihood = 0, 0
    bar = ChargingBar('Computing people data', max=len(envs))
    for ground_truth, actions, id_ in zip(envs, people_acts, ids):
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        likelihood_seq, ll_seq = 0, 0
        if id_[0] not in assignment.keys():
            print('{}: Not in the keys.'.format(id_))
        else:
            r = assignment[id_[0]][id_[1]]
        
            for act in actions:
                unobserved_nodes = env.get_unobserved_nodes()
                unobserved_node_labels = [node.label for node in unobserved_nodes]
                feature_values = env.get_node_feature_values(unobserved_nodes, 
                                                             softmax_feats,
                                                             softmax_norm_feats)
                dot_product = np.dot(weights[r-1], feature_values.T)
                if id_[1] < 20:
                    temp = train_temps[id_[0]]
                else:
                    temp = test_temps[id_[0]]
                softmax_dot = softmax2(dot_product, temp)
                ind = unobserved_node_labels.index(act)
                likelihood_seq += softmax_dot[ind]
                ll_seq += np.log(softmax_dot[ind])
            
                env.node_map[act].observe()
            likelihood += likelihood_seq
            log_likelihood += ll_seq
            total_len += len(actions)
        bar.next()
    bar.finish()
    
    mean_lik = likelihood / total_len
    geo_mean_lik = np.exp(log_likelihood / total_len)
    return mean_lik, geo_mean_lik

    
def compute_log_likelihood(envs, pipeline, people_acts, formula, epsilon_formula, 
                           num_formula, lik_matrix, verbose):
    """
    Similar to compute_score_people but computes the likelihood for belonging
    to cluster num_formula for each datapoint in the sequence of (envs, people_acts) 
    pairs given the Epsilon as an argument.
    
    Likelihood is P(d|Z=formula, Epsilon) so the probability of the description 
    given b formula under(envs, pipeline) serving as d and Epsilon being
    the epsilon_formula.
    
    Saves the likelihood for each datapoint in a matrix where num_formula
    defines the row to save the data in.
     
    Parameters
    ----------
    envs : [ [ int ] ]
        List of environments encoded as rewards hidden under the nodes; index of
        the reward corresponds to the id of the node in the Mouselab MDP
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes. For each rollout.
    people_acts : [ [ int ] ]
        List of action sequences taken by people in each consecutive environment
    formula : str
        Procedural formula
    epsilon_formula : float
        Epsilon for the action model discovered for the tested formula
    num_formula : int
        Tested formula's number; equivalently, the number of the cluster and its
        formula
    lik_matrix : np.array
        lik_matrix.shape = (num_formulas, num_datapoints)
        Matrix with the likelihoods for each datapoint under consecutive cluster
        formulas
        
    Returns
    -------
    lik_matrix : np.array
        lik_matrix.shape = (num_formulas, num_datapoints)
        Matrix with the likelihoods for each datapoint under consecutive cluster
        formulas
    """
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    num = -1
    if verbose: bar = ChargingBar('Computing log-lik for cluster {}'.format(num_formula), max=len(envs))
    for ground_truth, actions in zip(envs, people_acts):
        num += 1
        ll_allowed, ll_not_allowed = 0, 0
        epsilon_power, _1_epsilon_power = 0, 0
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        new_iter = True
        formulas = formula.split('\n\nOR\n\n')
        strategies = []
        for f in formulas:
            strategies.append(ProceduralStrategy(f))
            
        for action in actions:
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
                        
            if new_iter:
                applicable_strategies = [s for s in strategies 
                                         if s.applies(env, unobserved_node_labels)]
            if applicable_strategies == []:
                #No planning strategy
                allowed_acts = [0]
            else:
                allowed_acts = []
            untils = {s: s.until_condition_satisfied(env, unobserved_node_labels) 
                      for s in applicable_strategies}
            
            if any(list(untils.values())):
                applicable_strategies = [s for s in applicable_strategies 
                                         if untils[s]]
                try:
                    for s in applicable_strategies:
                        s.next()
                        if not(s.block_strategy(env, unobserved_node_labels)):
                             allowed_acts += return_allowed_acts(s.current, 
                                                                 env,
                                                                 unobserved_node_labels)
                        else:
                            allowed_acts = [0]
                        s.step()
                except:
                    allowed_acts = [0]
            else:
                for s in applicable_strategies:
                    if not(s.block_strategy(env, unobserved_node_labels)):
                        allowed_acts += return_allowed_acts(s.current, 
                                                            env,
                                                            unobserved_node_labels)
                    else:
                        allowed_acts = [0]
                    s.step()
            
            if formula == "None": allowed_acts = []
            allowed_acts = list(set(allowed_acts))
            allowed_distr = {a: 1/float(len(allowed_acts)) for a in allowed_acts}
            not_allowed_acts = [a for a in env.node_map.keys()
                                if a not in allowed_acts]
            not_allowed_distr = {a: 1/float(len(not_allowed_acts)) 
                                 for a in not_allowed_acts}
            if action in allowed_acts:
                ll_allowed += np.log(allowed_distr[action])
                _1_epsilon_power += 1
            else:
                ll_not_allowed += np.log(not_allowed_distr[action])
                epsilon_power += 1
            
            env.node_map[action].observe()
            new_iter = False
            
        if verbose: bar.next()
        if epsilon_power == 0:
            log_likelihood = ll_allowed
        elif _1_epsilon_power == 0:
            log_likelihood = ll_not_allowed
        else:
            log_likelihood = epsilon_power*np.log(epsilon_formula) + ll_not_allowed + \
                             _1_epsilon_power*np.log((1-epsilon_formula)) + ll_allowed
                             
        lik_matrix[num_formula][num] = np.exp(log_likelihood)
    if verbose: bar.finish()
    
    return lik_matrix
    
def compute_all_data_log_likelihood(weights, lik_matrix):
    """
    Computes the dot product of two arguments. Here, used for computing likelihood.
    
    Similar to compute_log_likelihood bbut computes the likelihood for each
    datapoint in the sequence of (envs, people_acts) pairs in general
    
    The likelihood is SUM_i=1^K 1/K * P(d|Z=i, Epsilon) so the sum of prior for
    belonging to cluster i (1/K) multiplied by the posterior.
     
    Parameters
    ----------
    weights : np.array
        weights.shape = (1, num_formulas)
        Matrix os weights to multiply likelihoods by.
    lik_matrix : np.array
        lik_matrix.shape = (num_formulas, num_datapoints)
        Matrix with the likelihoods for each datapoint under consecutive cluster
        formulas
        
    Returns
    -------
    data_log_liks : [ float ]
        List of likelihoods for each of the datapoints whose partial likelihoods 
        compute for each of the clusters is saved in lik_matrix.
    """
    data_liks = np.dot(np.transpose(lik_matrix), weights)
    data_log_liks = [np.log(d) for d in data_liks]
    return data_log_liks

def compute_scores(formula, weights, pipeline, softmax_features, 
                   softmax_normalized_features, envs, people_acts, ids, freq, verbose=True):
    """
    Extract human data belonging to a particular cluster found with the EM 
    algorithm.

    Parameters
    ----------
    (See compute_score_formula_rollouts or compute_score_softmax_rollouts)
    formula : str
    weights : [ float ]
    pipeline : [ ( Trial, function ) ]
    softmax_features : [ str ]
    softmax_normalized_features : dict
        str : float
    (See load_participant_data)
    envs : [ [ int ] ]
    people_acts : [ [ int ] ]
    ids : [ [ int ] ]
    freq: { int : float }
        
    Returns
    -------
    mean_ll_score : float
        Mean likelihood for rollouts under the softmax policy and the policy
        induced by the procedural description
    mean_opt_score : float
        Mean propotrion of optimal actions for rollouts under the same policies
        as above
    people_ll_score : float
        Likelihood of the human data under the policy induced by the procedural 
        description 
    ll_ppl : float
        Log likelihood for human data under the policy induced by the procedural
        formula
    opt_score_ppl : float
        Proportion of actions in the human data taken according to the allowed 
        actions of the formula-induced policy
    epsilon : float
         Measure of fit of the procedural description with respect to human data
    opt_act_score : float
        Average likelihood per planning operation over average optimal likelihood
        per planning operation when optimal is computed as though all (state, action)
        pairs agreed with the formula 
    results : str
        String that details all the computed statistcs and which is later passed
        into a text document that details the quality of EM clustering
    """
    res = compute_score_formula_rollouts(formula=formula, 
                                         W=weights, 
                                         pipeline=pipeline,
                                         softmax_features=softmax_features,
                                         softmax_norm_features=softmax_normalized_features,
                                         verbose=verbose)
    ll_soft, ll_greedy, opt_score, mean_len = res
    res = compute_score_softmax_rollouts(formula=formula, 
                                         W=weights, 
                                         pipeline=pipeline,
                                         softmax_features=softmax_features,
                                         softmax_norm_features=softmax_normalized_features,
                                         verbose=verbose)
    ll, opt_score2, mean_len2 = res
    res = compute_score_people(tuple(tuple(e) for e in envs), 
                                        pipeline=(tuple(pipeline[0][0]), pipeline[0][1]), 
                                        people_acts=tuple(tuple(a) for a in people_acts), 
                                        formula=formula,
                                        verbose=verbose)
    ll_ppl, opt_score_ppl, ml_ppl, epsilon, opt_act_score, m_lik, geo_m_lik = res
    
    with open('./data/v1.0_strategies.pkl', 'rb') as f:
        assignment = pickle.load(f)
    assignment[103] = [22, 31, 31, 31, 31, 31, 12, 6, 6, 6, 31, 31, 6, 6, 6, 6, 6] + \
                      [ 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 76]
                      
    with open('./data/microscope_weights.pkl', 'rb') as f:
        wgts = pickle.load(f)
    with open('./data/microscope_features.pkl', 'rb') as f:
        fts = pickle.load(f)
    
    res = compute_likelihood_microscope(envs=tuple(tuple(e) for e in envs), 
                                        pipeline=(tuple(pipeline[0][0]), pipeline[0][1]), 
                                        people_acts=tuple(tuple(a) for a in people_acts),
                                        weights=wgts,
                                        softmax_feats=fts,
                                        softmax_norm_feats=softmax_normalized_features,
                                        ids=ids,
                                        assignment=assignment)
    micro_m_lik, micro_geo_m_lik = res
                                             
    res = compute_likelihood_random(envs=tuple(tuple(e) for e in envs), 
                                    pipeline=(tuple(pipeline[0][0]), pipeline[0][1]), 
                                    people_acts=tuple(tuple(a) for a in people_acts))
    rand_m_lik, rand_geo_m_lik = res
    
    lik_imp = m_lik / rand_m_lik
    geo_lik_imp = geo_m_lik / rand_geo_m_lik
    micro_lik_imp = micro_m_lik / rand_m_lik
    micro_geo_lik_imp = micro_geo_m_lik / rand_geo_m_lik
    
    results = ''
    results += "\nFORMULA\n\nLL soft: {}\nLL greedy: {}\n".format(ll_soft, ll_greedy) + \
               "Score LL soft: {}\nScore LL greedy: {}\n".format(
                                                    np.exp(ll_soft/(ITERS*mean_len)),
                                                    np.exp(ll_greedy/(ITERS*mean_len))) + \
               'Mean len (formula): {}\nOpt score (formula): {}\n\n\n'.format(mean_len,
                                                                              opt_score)
    results += 'SOFTMAX\n\nLL: {}\nScore LL: {}\n'.format(ll, 
                                                          np.exp(ll/(ITERS*mean_len2))) + \
               'Opt score (softmax): {}\nMean len (softmax): {}\n\n\n'.format(opt_score2,
                                                                              mean_len2)
    mean_ll_score = (np.exp(ll_soft/(ITERS*mean_len)) + np.exp(ll/(ITERS*mean_len2)))/2
    mean_opt_score = (opt_score+opt_score2)/2
    results += 'MEAN\n\nLL: {}\nScore LL: {}\n'.format((ll_soft+ll)/2, 
                                                        mean_ll_score) + \
               'Opt score: {}\nMean len: {}\n\n\n'.format(mean_opt_score,
                                                          (mean_len+mean_len2)/2)
    people_ll_score = np.exp(ll_ppl/(len(envs)*ml_ppl))
    results += 'PEOPLE\n\nLL: {}\nScore LL: {}\n'.format(ll_ppl, people_ll_score) + \
               'Opt score: {}\nMean len: {}\nEpsilon: {}\n'.format(opt_score_ppl, 
                                                                   ml_ppl, epsilon) + \
               'Opt action score: {}\n\n\n'.format(opt_act_score) + \
               'COMPARISON\n\nMean lik: {}\nMicroscope: {}\n'.format(m_lik, micro_m_lik) + \
               'Random: {}\nImprovement: {} vs {}'.format(rand_m_lik, lik_imp, micro_lik_imp) + \
               '\n\nGeo lik: {}\nMicroscope: {}\n'.format(geo_m_lik, micro_geo_m_lik) + \
               'Random: {}\n'.format(rand_geo_m_lik) + \
               'Improvement: {} vs {}\n\n\n'.format(geo_lik_imp, micro_geo_lik_imp)
              
    results += 'SIZE: {}'.format(freq)
           
    if verbose: print(results)
      
    return mean_ll_score, mean_opt_score, people_ll_score, ll_ppl, opt_score_ppl, \
           epsilon, opt_act_score, results

if __name__ == "__main__":
    cwd = os.getcwd()

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', '-e',
                        type=str,
                        help="Identifier of the experiment to interpret.")
    parser.add_argument('--num_strategies', '-s',
                        type=int,
                        help="Number of strategies employed by the participants.")
    parser.add_argument('--num_participants', '-p',
                        type=int,
                        help="Number of participants whose data to take into account.",
                        default=0)
    parser.add_argument('--strategy_num', '-c',
                        type=int,
                        help="Number of the strategy to interpret.")
    parser.add_argument('--num_demos', '-n', 
                        type=int,
                        help="How many demos to use for interpretation.")

    args = parser.parse_args()
    
    #formula_path = cwd+'/interprets/'
    
    #file_name = args.formula
    #print("Retrieving the formula...")
    #with open(formula_path + file_name, 'rb') as handle:
    #    dict_object = pickle.load(handle)
    #formula = list(dict_object.keys())[0]
    #print('Done')
    
    formula = 'among(st, act, lambda st, act: not(is_observed(st, act)), lambda st, act, lst: has_parent_highest_value(st, act, lst)) AND NEXT True UNTIL IT STOPS APPLYING\n\nLOOP FROM among(lambda st, act: not(is_observed(st, act)), lambda st, act, lst: has_parent_highest_value(st,act,lst))'
    
    envs, action_seqs, pt = load_participant_data(exp_id=args.experiment_id,
                                                 num_clust=args.num_strategies,
                                                 num_part=args.num_participants,
                                                 clust_id=args.strategy_num)
    
    res = load_EM_data(exp_num=args.experiment_id, 
                       num_strategies=args.num_strategies,
                       num_participants=args.num_participants,
                       strategy_num=args.strategy_num,
                       num_simulations=args.num_demos)
    pipeline, weights, features, normalized_features = res

    compute_scores(formula=formula, 
                   weights=weights, 
                   pipeline=pipeline,
                   envs=envs,
                   people_acts=action_seqs,
                   softmax_features=features,
                   softmax_normalized_features=normalized_features,
                   ids=pt)
