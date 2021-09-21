from interpret_procedure import wrapper_interpret_human_data, \
                                wrapper_compute_procedural_formula
from evaluate_procedure import *
from interpret_evaluate_pipeline import simplify_and_optimize, wrapper_evaluate_procedural_formula, count_predicates
from RL2DT.hyperparams import *
from hyperparams import ALLOWED_PREDS, REDUNDANT_TYPES
from formula_procedure_transformation import sizeBig
from progress.bar import ChargingBar
import argparse
import pickle
import os
import numpy as np
from interpret_formula import load_EM_data
from RL2DT.strategy_demonstrations import make_modified_env as make_env
from scipy.special import softmax
from IHP.modified_mouselab import TrialSequence

def load_participant_data(exp_id, num_clust, num_part, clust_id):
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
              str(num_clust) + '_' + str(num_part) + '.pkl', 'rb') as handle:
        dict_object = pickle.load(handle)
    dct = dict_object[0]
    envs = [data[0] for data in dct[clust_id]]
    action_seqs = [data[1] for data in dct[clust_id]]
    part_trial = [data[-1] for data in dct[clust_id]]
    return envs, action_seqs, part_trial

def wrapper_evaluate_procedural_formula(**kwargs):
    """
    Wrapper function to evaluate the input procedural description.
    
    Human data is loaded and then the formula is evaluated against this data.
        
    Returns
    -------
    res : tuple
        Tuple containing evaluation statistics and a string with these statistics. 
        See compute_scores in evaluate_procedure.py
    """
    print("\n\n\n\n\n                                " +\
          " EVALUATING THE PROCEDURAL FORMULA: \n\n\n\n\n")
    envs, action_seqs, pt = load_participant_data(exp_id=kwargs['experiment_id'],
                                              num_clust=kwargs['num_strategies'],
                                              num_part=kwargs['num_participants'],
                                              clust_id=kwargs['strategy_num'])
    
    res = compute_likelihood_people(tuple(tuple(e) for e in envs), 
                                        pipeline=(tuple(kwargs['pipeline'][0][0]), kwargs['pipeline'][0][1]), 
                                        people_acts=tuple(tuple(a) for a in action_seqs), 
                                        formula=kwargs['raw_formula'],
                                        formula_num=i,
                                        verbose=True)
                                        
    res2 = compute_likelihood_microscope(tuple(tuple(e) for e in envs), 
                                             pipeline=(tuple(kwargs['pipeline'][0][0]), kwargs['pipeline'][0][1]), 
                                             people_acts=tuple(tuple(a) for a in action_seqs),
                                             weights=kwargs['weights'],
                                             softmax_feats=kwargs['softmax_features'],
                                             softmax_norm_feats=kwargs['softmax_normalized_features'],
                                             microscope=kwargs['microscope'],
                                             ids=pt)
                                             
    lik_random = compute_likelihood_random(tuple(tuple(e) for e in envs), 
                                              pipeline=(tuple(kwargs['pipeline'][0][0]), kwargs['pipeline'][0][1]), 
                                              people_acts=tuple(tuple(a) for a in action_seqs))
    
    return (res[0], res[1])+res2+(lik_random,)

#####################################################################################################################
############# Average likelihood per planning operation computation for DNF2LTL, hand-made method and a random model
##################################################################################################################### 
## ad-hoc file loading    
with open('/home/julian/Pobrane/si_strategy_map.pkl', 'rb') as f:
    sism = pickle.load(f)
relevant_strategies = list(sism.values())
with open('/home/julian/Pobrane/strategy_si_map.pkl', 'rb') as f:
    ssim = pickle.load(f)
with open('/home/julian/Pobrane/v1.0_strategies.pkl', 'rb') as f:
    assignment = pickle.load(f)
assignment[103] = [22, 31, 31, 31, 31, 31, 12, 6, 6, 6, 31, 31, 6, 6, 6, 6, 6, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 76]
with open('/home/julian/Pobrane/inferred_strategies/inferred_strategies/increasing_variance_test/temperatures.pkl', 'rb') as f:
    test_temps = pickle.load(f)
with open('/home/julian/Pobrane/increasing_variance_training/temperatures.pkl', 'rb') as f:
    train_temps = pickle.load(f)
EPSILON = [0.007867132867132868, 0.06832210998877665, 0.0, 0.16647465437788017, 0.3928354584092289, 0.0, 0.29949238578680204, 0.0, 0.5159268929503916, 0.07969410344133629]

def softmax2(x, temp):
    e_x = np.exp((x - np.max(x))/ temp)
    return e_x / e_x.sum()
    
from functools import lru_cache
@lru_cache()
def compute_likelihood_people(envs, pipeline, people_acts, formula, formula_num, verbose):
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    total_len = 0
    total_likelihood = 0
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
                total_likelihood += (1-EPSILON[formula_num]) * allowed_distr[action]
            else:
                total_likelihood += EPSILON[formula_num] * not_allowed_distr[action]
            
            env.node_map[action].observe()
            new_iter = False
        if verbose: bar.next()
    if verbose: bar.finish()
    
    return total_likelihood, total_len
    
def compute_likelihood_microscope(envs, pipeline, people_acts, weights, softmax_feats, softmax_norm_feats, microscope, ids):
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    optimal_acts = 0
    total_len = 0
    likelihood = 0
    bar = ChargingBar('Computing people data', max=len(envs))
    for ground_truth, actions, id_ in zip(envs, people_acts, ids):
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        likelihood_seq = 0
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
            
                env.node_map[act].observe()
            likelihood += likelihood_seq
            si_cluster = ssim[r]
            microscope[si_cluster].append(likelihood_seq)
            elements[si_cluster].append(id_)
            total_len += len(actions)
        bar.next()
    bar.finish()
    return likelihood, total_len, microscope, elements
    
def compute_likelihood_random(envs, pipeline, people_acts):
    envs = [list(e) for e in envs]
    pipeline = [(list(pipeline[0]), pipeline[1])]
    people_acts = [list(p) for p in people_acts]
    
    optimal_acts = 0
    likelihood = 0
    bar = ChargingBar('Computing people data', max=len(envs))
    for ground_truth, actions in zip(envs, people_acts):
        env = TrialSequence(1, pipeline, ground_truth=[ground_truth]).trial_sequence[0]
        likelihood_seq = 0
        
        for act in actions:
            unobserved_nodes = env.get_unobserved_nodes()
            unobserved_node_labels = [node.label for node in unobserved_nodes]
            random_distr = softmax([1./len(unobserved_nodes) for _ in range(len(unobserved_nodes))])
            ind = unobserved_node_labels.index(act)
            likelihood_seq += random_distr[ind]
            
            env.node_map[act].observe()
            
        likelihood += likelihood_seq
        bar.next()
    bar.finish()
    return likelihood
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', '-e',
                        type=str,
                        help="Identifier of the experiment to interpret.")
    parser.add_argument('--num_strategies', '-s',
                        type=int,
                        help="Number of strategies employed by the participants.")
    parser.add_argument('--num_participants', '-p',
                        type=int,
                        help="Number of participants whose data was taken into account.",
                        default=0)
    parser.add_argument('--demo_path', '-dp',
                        type=str,
                        help="Path to the file with the demonstrations.", 
                        default='')
    parser.add_argument('--num_demos', '-n', 
                        type=int,
                        help="How many demos to use for interpretation.")
    parser.add_argument('--elbow_choice', '-elb',
                        choices={'automatic','manual'},
                        help="Whether to find the candidates for the number of " \
                            +"clusters automatically or use the " \
                            +"candidate_clusters parameter.",
                        default='automatic')
    parser.add_argument('--mean_reward', '-mr',
                        type=float,
                        help="Mean reward the interpretation will aspire to.",
                        default=None)
    parser.add_argument('--expert_reward', '-er',
                        type=float,
                        help="Mean reward of the optimal strategy for the problem.")
    parser.add_argument('--candidate_clusters', '-cl',
                        type=int, nargs='+',
                        help="The candidate(s) for the number of clusters in " \
                            +"the data either to consider in their entirety " \
                            +"or to automatically choose from.",
                        default=NUM_CLUSTERS)
    parser.add_argument('--num_candidates', '-nc',
                        type=int,
                        help="The number of candidates for the number of " \
                            +"clusters to consider.",
                        default=NUM_CANDIDATES)
    parser.add_argument('--name_dsl_data', '-dsl', 
                        type=str,
                        help="Name of the .pkl file containing input demos turned " \
                            +"to binary vectors of predicates in folder demos", 
                        default=None)
    parser.add_argument('--interpret_size', '-i',
                        type=int,
                        help="Maximum depth of the interpretation tree",
                        default=MAX_DEPTH)
    parser.add_argument('--max_divergence', '-md',
                        type=float,
                        help="How close should the intepration performance in " \
                            +"terms of the reward be to the policy's performance",
                        default=MAX_DIVERGENCE)
    parser.add_argument('--tolerance', '-t',
                        type=float,
                        help="What increase in the percentage of the expected " \
                             "expert reward a formula is achieving is " \
                             "considered significant when comparing a two of them.",
                        default=TOLERANCE)
    parser.add_argument('--num_rollouts', '-rl',
                        type=int,
                        help="How many rolouts to perform to compute mean " \
                            +"return per environment",
                        default=NUM_ROLLOUTS)
    parser.add_argument('--samples', '-sm',
                        type=float,
                        help="How many samples/in what ratio to sample from clusters",
                        default=SPLIT)
    parser.add_argument('--info', '-f',
                        type=str,
                        help="What to add to the name of all the output files",
                        default='')

    args = parser.parse_args()
    
    total_max_lik = []
    num_params_CNF2LTL = 0
    
    total_microscope_lik = 0
    microscope = {i+1: [] for i in range(len(relevant_strategies))}
    elements = {i+1: [] for i in range(len(relevant_strategies))}
    len_ = []
    
    total_random_lik = 0
    
    for i in range(args.num_strategies):
        if ((args.num_strategies == 18 and i == 16) or (args.num_strategies == 20 and i == 14)):
            continue
    
        print('\n' * 5 + ' ' * 40 + 'CLUSTER {}'.format(i) + '\n'*5)
    
        res = wrapper_interpret_human_data(experiment_id=args.experiment_id,
                                       num_strategies=args.num_strategies,
                                       num_participants=args.num_participants,
                                       num_trajs=args.num_demos,
                                       strategy_num=i,
                                       max_divergence=args.max_divergence, 
                                       size=args.interpret_size, 
                                       tolerance=args.tolerance, 
                                       num_rollouts=args.num_rollouts, 
                                       num_samples=args.samples, 
                                       num_candidates=args.num_candidates, 
                                       candidate_clusters=args.candidate_clusters, 
                                       name_dsl_data=args.name_dsl_data, 
                                       demo_path=args.demo_path,
                                       elbow_method=args.elbow_choice,
                                       mean_rew=args.mean_reward,
                                       expert_rew=args.expert_reward,
                                       info=args.info)
        formula, pred_matrix, demos, pipeline, weights, features, normalized_features = res
    
        res = wrapper_compute_procedural_formula(formula=formula,
                                                 pred_matrix=pred_matrix,
                                                 demos=demos,
                                                 experiment_id=args.experiment_id,
                                                 num_strategies=args.num_strategies,
                                                 num_participants=args.num_participants,
                                                 strategy_num=i,
                                                 pipeline=pipeline)
        proc_formula, raw_proc_formula, c = res
        
        proc_parts = proc_formula.split('\n\nOR\n\n')
        raw_proc_parts = raw_proc_formula.split('\n\nOR\n\n')
        new_proc_formula, raw_new_proc_formula = '', ''
        best_lmarglik = -np.inf
        for proc_formula, raw_proc_formula in zip(proc_parts, raw_proc_parts):
            num_params_CNF2LTL += count_predicates(proc_formula, raw_proc_formula)
            res = simplify_and_optimize(formula=proc_formula, formula_raw=raw_proc_formula,
                                        exp_id=args.experiment_id,
                                        num_str=args.num_strategies,
                                        num_part=args.num_participants,
                                        str_num=i,
                                        pipeline=pipeline)
            part_proc_formula, part_raw_proc_formula, lmarglik = res
            if lmarglik > best_lmarglik:
                new_proc_formula = part_proc_formula
                raw_new_proc_formula = part_raw_proc_formula
                best_lmarglik = lmarglik
        c = sizeBig(new_proc_formula)
        #b_l = new_proc_formula.split('\n\nLOOP FROM ')
        #c = size(b_l[0]) + size(b_l[1]) if len(b_l) == 2 else size(b_l[0])
        proc_formula = new_proc_formula
        raw_proc_formula = raw_new_proc_formula
    
        print('\n\nOPTIIMIZED PROCEDURAL FORMULA:\n\n{}'.format(proc_formula) + \
              '\n\nComplexity: {}'.format(c))
        
        with open('./data/microscope_weights.pkl', 'rb') as f:
            wgts = pickle.load(f)
        with open('./data/microscope_features.pkl', 'rb') as f:
            fts = pickle.load(f)
    
        res = wrapper_evaluate_procedural_formula(experiment_id=args.experiment_id,
                                                  num_strategies=args.num_strategies,
                                                  num_participants=args.num_participants,
                                                  strategy_num=i,
                                                  raw_formula=raw_proc_formula,
                                                  weights=wgts,
                                                  pipeline=pipeline,
                                                  softmax_features=fts,
                                                  softmax_normalized_features=normalized_features,
                                                  microscope=microscope,
                                                  elements=elements)
        total_random_lik += res[-1]
        res = res[:-1]                                          
        total_max_lik += [res[0]]
        total_microscope_lik += res[2]
        len_ += [res[3]]
        microscope = res[-2]
        elements = res[-1]
        print(elements)
        
    new_elements = {i: [0]*30 for i in range(180)}
    for el, where in elements.items():
        for (part_id, place) in where:
            new_elements[part_id][place] = sism[el]
    print()
    def ind_0(el):
        if 0 in el:
            return el.index(0)
        else:
            return len(el)
    new_elements = {k: v[:ind_0(v)] for k,v in new_elements.items() if v != [0] * 30}
    
        
    print('\n\n\nTotal likelihood AUTOMATIC: {}'.format(sum(total_max_lik)))
    print('Total likelihood HAND-MADE: {}'.format(total_microscope_lik))
    print('Total likelihood RANDOM: {}'.format(total_random_lik))
    plan_op = sum([len(microscope[k]) for k in microscope.keys()])
    print('\nWe have {} planning operations'.format(plan_op))
    percentages = {key: len(microscope[key])/plan_op for key in microscope.keys()}
    print('\nSanity-check usage of each strategy: {}'.format(percentages))
    print()
    print('Number of all the planning operations: {}'.format(sum(len_)))
    print()
    lik_per_operation_ihp = sum(total_max_lik) / sum(len_)
    lik_per_operation_microscope = total_microscope_lik / sum(len_)
    lik_per_operation_random = total_random_lik / sum(len_)
    print('\nLikelihood per planning operation AUTOMATIC: {}'.format(lik_per_operation_ihp))
    print('Likelihood per planning operation HAND-MADE: {}'.format(lik_per_operation_microscope))
    print('Likelihood per planning operation RANDOM: {}'.format(lik_per_operation_random))
    print()
    print('Likelihood ratio per planning operation wrt random planning AUTOMATIC: {}'.format(lik_per_operation_ihp / lik_per_operation_random))
    print('Likelihood ratio per planning operation wrt random planning HAND-MADE: {}'.format(lik_per_operation_microscope / lik_per_operation_random)) 
