from interpret_procedure import wrapper_interpret_human_data, \
                                wrapper_compute_procedural_formula
from IHP.learning_utils import create_dir
from RL2DT.hyperparams import *
from translation import alternative_procedures2text as translate
import argparse
import pickle
import os
from toolz import curry
import copy
from formula_visualization import dnf2conj_list as d2l
from geneticalgorithm import geneticalgorithm as ga
from evaluate_procedure import compute_scores, load_participant_data, \
                               compute_score_people, compute_log_likelihood, \
                               compute_all_data_log_likelihood
from formula_procedure_transformation import size, sizeBig
import numpy as np
import sys


def represent_modular(formula, raw_formula):
    """
    Represent the input procedural formula in a list of steps.
    
    Parameters
    ----------
    formula : str
        Procedural formula 
    formula_raw : str
        Procdeural formula written in a callable form
        
    Returns
    -------
    modular_representation : [ dict ]
        dict
            conjunction : str
                Conjunction of predicates encoding a step of the procedural formula
            until : str
                Until condition for the step 
            unless : str
                Unless condition for the step
            conjunction_raw : str
                Same as above but wirtten in callable form 
            until_raw : str
                Same as above but wirtten in callable form 
            unless_raw : str
                Same as above but wirtten in callable form 
            go_to : str
                Conjunction to return to after completing all the steps
            go_to_raw : str
                Same as above but wirtten in callable form
    ids : [ str ]
        List of unique predicates uesed in the procedural description
    """
    modular_representation = []
    ids = []
    def id_predicates(conj):
        preds = d2l(conj, paran=False)[0]
        for p in preds:
            if p not in ids:
                ids.append(p)
    
    parts = formula.split(' AND NEXT ')
    parts_raw = raw_formula.split(' AND NEXT ')
    go_to, go_to_raw = '', ''
    if 'LOOP' in parts[-1]:
        last_el, go_to = parts[-1].split('\n\n')
        last_el_raw, go_to_raw = parts_raw[-1].split('\n\n')
        parts[-1] = last_el
        parts_raw[-1] = last_el_raw
        
    for index, (p, pr) in enumerate(zip(parts, parts_raw)):
        unless_p, unless_pr = '', ''
        first_p, first_pr = p, pr
        until_p, until_pr = '', ''
        if 'UNLESS' in p:
            first_p, unless_p = p.split(' UNLESS ')
            first_pr, unless_pr = pr.split(' UNLESS ')
        if 'UNTIL' in p:
            first_p, until_p = first_p.split(' UNTIL ')
            first_pr, until_pr = first_pr.split(' UNTIL ')
        if first_pr[:2] == '((':
            first_pr = first_pr[2:]
        if first_pr[-2:] == '))' and first_pr[-5:] != 'act))' and \
           first_pr[-3:] != ' ))':
            first_pr = first_pr[:-2]
        elif first_pr[-3:] == ')) ' and first_pr[-5:] != 'act)) ' and \
             first_pr[-4:] != ' )) ':
            first_pr = first_pr[:-3]
        id_predicates(first_p)
        
        dict_ = {'conjunction': first_p, 'until': until_p, 'unless': unless_p,
                 'conjunction_raw': first_pr, 'until_raw': until_pr, 
                 'unless_raw': unless_pr}
        modular_representation.append(dict_.copy())
        
    modular_representation.append({'go_to': go_to, 'go_to_raw': go_to_raw})
    return modular_representation, ids

def count_predicates(formula, raw_formula):
    """
    Determine the number of unique predicates appearing in the input procedural
    formula.
    
    Parameters
    ----------
    formula : str
        Procedural formula 
    formula_raw : str
        Procdeural formula written in a callable form
        
    Returns
    -------
    num : int
        Number of unique predicates in 'formula'
    """
    _, ids = represent_modular(formula, raw_formula)
    num = len(ids)
    return num

def select_subformula(modular_representation, predicate_ids, binary_vector):
    """
    Output a procedural subformula of a formula represented in modular_representation
    by selecting only the predicates which were not removed according to the
    binary_vector.
    
    Parameters
    ----------
    (See represent_modular)
    modular_representation : [ dict ]
    predicate_ids : [ str ]
    binary_vector : np.array
        vec.shape = (1, len(modular_representation)-1)
        Binary vector representing the steps included in a procedural formula
        divided into separate elements of modular_representation
    
    Returns
    -------
    formula : str
        Procedural subformula 
    formula_raw : str
        Procdeural subformula written in a callable form
    """
    formula, formula_raw = '', ''
    loop_exists = False
    go_to = modular_representation[-1]
    g = go_to['go_to']
    g_raw = go_to['go_to_raw']
    if g!= '': 
        g = g[10:] #remove LOOP FROM
        g_raw = g_raw[10:] #remove LOOP FROM
    g_until, g_until_raw = '', ''
    prev = ''
    if 'UNLESS' in g:
        g, g_until = g.split(' UNLESS ')
        g_raw, g_until_raw = g_raw.split(' UNLESS ')
        g_until = ' UNLESS ' + g_until
        g_until_raw = ' UNLESS ' + g_until_raw
    last_until = False
        
    for ind, mod in enumerate(modular_representation[:-1]):
        conj = d2l(mod['conjunction'], paran=False)[0]
        conj_raw = d2l(mod['conjunction_raw'], paran=False)[0]
        current, current_raw = '', ''
        loop_formula = False

        if mod['conjunction'] == g:
            loop_formula = True
            loop_exists = True
            goto, goto_raw = '', ''
        all_zero = True
        for c, cr in zip(conj, conj_raw):
            vector_index = predicate_ids.index(c)
            if binary_vector[vector_index] == 1:
                if all_zero:
                    current += c
                    current_raw += cr
                    if loop_formula:
                        goto += c
                        goto_raw += cr
                else:
                    current += ' and ' + c
                    current_raw += ' and ' + cr
                    if loop_formula:
                        goto += ' and ' + c
                        goto_raw += ' and ' + cr
                all_zero = False
                
        if all_zero: #whole conjunction removed = always True
            current += 'True'
            current_raw += 'True'
            if loop_formula:
                goto += 'True'
                goto_raw += 'True'

        if current != prev: #different predicate after AND NEXT than before
            formula += current
            formula_raw += current_raw
            last_until = False
            
            if mod['until'] != '':
                formula += ' UNTIL ' + mod['until']
                formula_raw += ' UNTIL ' + mod['until_raw']
                last_until = True
            
            if mod['unless'] != '':
                formula += ' UNLESS ' + mod['unless']
                formula_raw += ' UNLESS ' + mod['unless_raw']
                last_until = True
            
            formula += ' AND NEXT '
            formula_raw += ' AND NEXT '
            
        else:
            len_until = 2 if ' or ' in mod['until'] else 1 if mod['until'] != '' else 0
            len_unless = 2 if ' or ' in mod['unless'] else 1 if mod['unless'] != '' else 0
        
        if current == 'True' and current!= prev and \
           'STOPS APPLYING' in mod['until']: #instruction denoting clicking 
                                             #randomly and terminating; the rest
                                             #of the procedure isn't important
            loop_exists = False
            break
        prev = current
        
    formula = formula[:-10]
    formula_raw = formula_raw[:-10]
    
    if loop_exists:
        if current == goto:
            if g_until == '': 
                if not(last_until):
                    formula += ' UNTIL IT STOPS APPLYING'
                    formula_raw += ' UNTIL IT STOPS APPLYING'
            else:
                if not(last_until):
                    formula += ' UNTIL ' + g_until[8:]
                    formula_raw += ' UNTIL ' + g_until_raw[8:]
        else:
            formula += '\n\nLOOP FROM ' + goto + g_until
            formula_raw += '\n\nLOOP FROM ' + goto_raw + g_until_raw
            
    return formula, formula_raw


def progress(count, total, status=''):
    """
    Function that displays a loading bar.
    
    Parameters
    ----------
    count : int
    total : int
    status : str (optional)
    """
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush() 

def greedy_check(function, dimension):
    """
    Search for the optimal (minimum) solution of a binary funciton 
                   'function' : {0,1}^dimension -> IR 
    by doing a greedy search.
    
    Start with a vector <1,...,1> and switch 1 to 0 if it improves the value of
    'function'. Repeat 'dimension' times to try all the possibilities.
    
    Parameters
    ----------
    function : callable
        Pythonic function that accepts a binary vector as an input and returns a
        real value
    dimension : int
        Dimension of 'function''s domain
    
    Returns
    -------
    vec : np.array
        vec.shape = (1, dimension)
        Greedy-optimal binary vector minimizing the value of 'function'
    best_val : float
        Greedy-minimal value
    """
    vec = np.ones(dimension)
    best_val = function(vec)
    print('\nInitial value: {}\n'.format(best_val))
    counter = 0
    for _ in range(len(vec)):
        for pos, _ in enumerate(vec):
            counter += 1
            progress(counter, (len(vec))**2, status="Greedy algorithm is running...")
            test_vec = vec.copy()
            test_vec[pos] = 0
            val = function(test_vec)
            print('\nVec: {}, val: {}'.format(test_vec, val))
            if val < best_val:
                best_val = val
                vec = test_vec
    print('\nGreedy vector found: {}\n\nwith value: {}\n'.format(vec, best_val))
    return vec, best_val

@curry
def quality_function(binary_vector, modular_rep, pred_ids, people_data, pipeline):
    """
    Evaluate the quality of a binary vector.
    
    Vector represents which steps of a procedural formula represented in modular_rep
    (divided by NEXT steps) are to be considered, and which not.
    
    Quality is the number of removed steps minus the log likelihood of the pruned 
    formula under people's actions (better vectors are the smaller ones and 
    those which fit the data well).
    
    Parameters
    ----------
    binary_vector : np.array
        vec.shape = (1, len(modular_rep)-1)
        Binary vector representing the steps included in a procedural formula
        divided into separate elements of modular_rep
    (See represent_modular)
    modular_rep : [ dict ]
    pred_ids : [ str ]
    people_data : ( [ [ int ] ], [ [ int ] ] )
        List of environments encoded as rewards hidden under the nodes; List of 
        action sequences taken by people in each consecutive environment
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes.
    
    Returns
    -------
    score : float
        Computed quality of the input vector
    """
    
    selected_formula, selected_formula_raw = select_subformula(modular_rep, 
                                                               pred_ids,
                                                               binary_vector)
    envs, action_seqs = people_data
    
    res = compute_score_people(envs=tuple(tuple(e) for e in envs), 
                               pipeline=(tuple(pipeline[0][0]), pipeline[0][1]), 
                               people_acts=tuple(tuple(a) for a in action_seqs), 
                               formula=selected_formula_raw,
                               verbose=False)
    ll_ppl, opt_score_ppl, ml_ppl, eps, opt_score_people, mn_lik, mn_g_lik = res
    
    score = ll_ppl - sum(binary_vector)
    return -score #genetic algorithm minimizes by default

def simplify_and_optimize(formula, formula_raw, exp_id, num_str, 
                          num_part, str_num, pipeline, info, algorithm='greedy'):
    """
    Prune the input formula to finds its simpler representation which achieves 
    higher or equal log likelihood.
    
    Parameters
    ----------
    formula : str
        Procedural formula 
    formula_raw : str
        Procdeural formula written in a callable form
    exp_id : str
        Identifier of the experiment from which the human data (used to evaluate
        the pruned formulas) comes from
    num_str : int
        Identifier of the model with 'num_str' strategies
    num_part : int
        Number of participants of exp_id whose data was considered
    str_num : int
        Identifier of the strategy
    pipeline : [ ( [ int ], function ) ]
        List of parameters used for creating the Mouselab MDP environments: 
        branching of the MDP and the reward function for specifying the numbers
        hidden by the nodes.
    algorithm : str (optional)
        Whether to perform 'ga' genetic algorithm optimization or use the 'greedy'
        method
    
    Returns
    -------
    new_formula : str 
        Best procedural subformula of the input procedural formula according to
        the marginal likelihood
    new_formula_raw : str
        Same formula as above written in a callable form
    ll_ppl : float
        Log likelihood of the policy indeced by the pruned procedural formula
    """
    assert algorithm in ['greedy', 'ga'], "algorithm needs to be 'greedy' or 'ga'" + \
        " not {}".format(algorithm)
    envs, action_seqs, _ = load_participant_data(exp_id=exp_id,
                                                 num_clust=num_str,
                                                 num_part=num_part,
                                                 clust_id=str_num,
                                                 info=info)
    p_data = (envs, action_seqs)
    modular_formula_rep, predicate_ids = represent_modular(formula, formula_raw)
    
    ga_vec_size = count_predicates(formula, formula_raw)
    ga_q_func = quality_function(modular_rep=modular_formula_rep,
                                 pred_ids=predicate_ids,
                                 people_data=p_data,
                                 pipeline=pipeline)
    ga_varbound=None #for Boolean type
    ga_params={'max_num_iteration': None,'population_size': 100, 
               'mutation_probability': 0.5, 'elit_ratio': 0.01, 
               'crossover_probability': 0.5, 'parents_portion': 0.3,
               'crossover_type':'uniform', 'max_iteration_without_improv': None}
    model=ga(function=ga_q_func,
             function_timeout=20,
             dimension=ga_vec_size,
             variable_type='bool',
             variable_boundaries=ga_varbound,
             convergence_curve=False,
             algorithm_parameters=ga_params)
    
    if algorithm == 'greedy':
        best_binary_vec, val = greedy_check(function=ga_q_func,
                                           dimension=ga_vec_size)
    elif algorithm == 'ga':
        print('Genetic algorithm is running...')
        model.run()
        best_binary_vec = model.best_variable
    
    new_formula, new_formula_raw = select_subformula(modular_formula_rep,
                                                     predicate_ids,
                                                     best_binary_vec)
    ll_ppl = -val + sum(best_binary_vec)
    
    return new_formula, new_formula_raw, ll_ppl


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
    envs, action_seqs, pt, freq = load_participant_data(exp_id=kwargs['experiment_id'],
                                                        num_clust=kwargs['num_strategies'],
                                                        num_part=kwargs['num_participants'],
                                                        clust_id=kwargs['strategy_num'],
                                                        info=kwargs['info'],
                                                        freqs=True)

    res = compute_scores(formula=kwargs['raw_formula'], 
                         weights=kwargs['weights'], 
                         pipeline=kwargs['pipeline'],
                         softmax_features=kwargs['softmax_features'],
                         softmax_normalized_features=kwargs['softmax_normalized_features'],
                         envs=envs,
                         people_acts=action_seqs,
                         ids=pt,
                         freq=freq)
    return res

def wrapper_interpret_cluster_to_procedural_description(**kwargs):
    """ 
    Wrapper function for creting procedural description of a human planning
    strategy evidenced by the demonstrations of that strategy. 
    
    The function finds a DNF formula first, transforms it into a procedural 
    program-like description, prunes it, evaluates the final description compared 
    to the human demonstrations residing in the cluster and returns relevant data 
    in a text file.
    
    Returns
    -------
    new_res : tuple
        Tuple containing evaluation statistics and a string with these statistics
        and the procedural description itself. (And the number of used features).
        It also returns the pipeline. See compute_scores in evaluate_procedure.py
    """
    print("\n\n\n\n\n                                " +\
          " CLUSTER {}: \n\n".format(kwargs['strategy_num']))
    res = wrapper_interpret_human_data(experiment_id=kwargs['experiment_id'],
                                       num_strategies=kwargs['num_strategies'],
                                       num_participants=kwargs['num_participants'],
                                       num_trajs=kwargs['num_trajs'],
                                       strategy_num=kwargs['strategy_num'],
                                       max_divergence=kwargs['max_divergence'], 
                                       size=kwargs['size'], 
                                       tolerance=kwargs['tolerance'], 
                                       num_rollouts=kwargs['num_rollouts'], 
                                       num_samples=kwargs['num_samples'], 
                                       num_candidates=kwargs['num_candidates'], 
                                       candidate_clusters=kwargs['candidate_clusters'], 
                                       name_dsl_data=kwargs['name_dsl_data'], 
                                       demo_path=kwargs['demo_path'],
                                       elbow_method=kwargs['elbow_method'],
                                       mean_reward=kwargs['mean_reward'],
                                       expert_reward=kwargs['expert_reward'],
                                       info=kwargs['info'])
    formula, pred_matrix, demos, pipeline, weights, feats, normalized_feats = res
    
    res = wrapper_compute_procedural_formula(formula=formula,
                                             pred_matrix=pred_matrix,
                                             demos=demos,
                                             pipeline=pipeline,
                                             **kwargs)
    proc_formula, raw_proc_formula, c = res
    
    proc_parts = proc_formula.split('\n\nOR\n\n')
    raw_proc_parts = raw_proc_formula.split('\n\nOR\n\n')
    
    num_params = len(feats) + 1 #for BIC and AIC, the number of optimized parameters
                                #cluster params and Epsilon
    new_proc_formula, raw_new_proc_formula = '', ''
    best_ll = -np.inf
    for proc_formula, raw_proc_formula in zip(proc_parts, raw_proc_parts):
        num_params += count_predicates(proc_formula, raw_proc_formula)
        res = simplify_and_optimize(formula=proc_formula, formula_raw=raw_proc_formula,
                                    exp_id=kwargs['experiment_id'],
                                    num_str=kwargs['num_strategies'],
                                    num_part=kwargs['num_participants'],
                                    str_num=kwargs['strategy_num'],
                                    pipeline=pipeline,
                                    info=kwargs['info'])
        part_proc_formula, part_raw_proc_formula, ll = res
        if ll > best_ll:
            new_proc_formula = part_proc_formula
            raw_new_proc_formula = part_raw_proc_formula
            best_ll = ll
    c = sizeBig(new_proc_formula)
    proc_formula = new_proc_formula
    raw_proc_formula = raw_new_proc_formula
    
    print('\n\nOPTIIMIZED PROCEDURAL FORMULA:\n\n{}'.format(proc_formula) + \
          '\n\nComplexity: {}'.format(c))
    
    res = wrapper_evaluate_procedural_formula(experiment_id=kwargs['experiment_id'],
                                              num_strategies=kwargs['num_strategies'],
                                              num_participants=kwargs['num_participants'],
                                              strategy_num=kwargs['strategy_num'],
                                              raw_formula=raw_proc_formula,
                                              weights=weights,
                                              pipeline=pipeline,
                                              softmax_features=feats,
                                              softmax_normalized_features=normalized_feats,
                                              info=kwargs['info'])
    _, _, _, _, _, _, _, evaluation_log = res
    strategy_log = 'STRATEGY {}/{}: {}\n\nComplexity: {}'.format(kwargs['strategy_num'],
                                                                 kwargs['num_strategies'],
                                                                 proc_formula, c)
    text = translate(proc_formula)
    strategy_log += '\n\nTranslation: {}\n'.format(text)
    strategy_log = '\n\n' + '-'*30 + '\n\n' + strategy_log
    strategy_log += evaluation_log
    none_ = 1 if 'None ' in proc_formula else 0
    formula_params = [raw_proc_formula, pipeline]
    
    new_res = (num_params, none_) + res[:-1] + (strategy_log, formula_params)
    return new_res
    
def save_strategies_and_evaluation(filepath, filename, data, append=True):
    """
    Modify or create a text file with procedural descriptions of human planning
    strategies. 
    
    The file also includes statistics connected to how well the descriptions 
    correspond to the demonstrations used in their creations and human data.

    Parameters
    ----------
    filepath : str
        Path to the file
    filename : str 
        Name of the file
    data : str
        String containing the descriptions and the statistics
    append : bool
        Whether to create a new file or add to an existing one
    """
    create_dir(filepath)
    if append:
        with open(filepath+'/'+filename, "a") as strategies:
            strategies.write(data)
            strategies.write("\n\n")
    else:
        with open(filepath+'/'+filename, "w") as strategies:
            strategies.write(data)
            strategies.write("\n\n")
            
def cluster_importance_weighting(exp_id, num_clust, num_p, info):
    """
    Establish the proportion of demonstrations each of the num_clust clusters 
    encoding a planning strategy from experiment exp_id covers. 

    Parameters
    ----------
    exp_id : str
        Name of the experiment for which the clusters of state-action pairs were
        created
    num_clust : int 
        Number of probabilsitic (EM) clusters
    num_p : int
        Number of participants whose data was considered for creating the clusters
        
    Returns
    -------
    reps : dict
        int : float
            Proportion of human demostrations assigned to cluster number int
    all_ : int
        Number of all human demonstrations/planning operations
    """
    cluster_folder = f"clustering/em_clustering_results/{exp_id}/{num_clust}_{num_p}{info}.pkl"
        
    with open(cluster_folder, 'rb') as handle:
        dict_object = pickle.load(handle)
    dct = dict_object[0]
    lens = {k: len(list([i[-1] for i in dct[k]])) for k in dct.keys()}
    all_ = sum(lens.values())
    reps = {k : v/all_ for k,v, in lens.items()}
    return reps, all_
    
def interpret_evaluate_clusters(log, **kwargs):
    """
    High-level function for computing the descriptions of EM clusters, pruning
    them, evaluating them compared to the demonstrations they were created from, 
    and compared to human data the EM clusters represent.
    
    Parameters
    ----------
    log : bool
        Whether to save the data on the computed descriptions in a text file
        
    Returns
    -------
    all_scores : dict
        likelihood_per_action : [ float ]
            Average likelihood of an action computed as a mean on the likelihood 
            of rollouts of the EM clusters' softmax policies under the descriptions,
            and the likelihood of rollouts of the descriptions under the softmax 
            policies. Computed for each cluster
        opt_score : [ float ] 
            Average proportion of times the descriptions/the sofotmax policies
            took an optimal action considering the softmax policies/the descriptions
            (optimal = highest probability). Computed for each cluster
        ppl_likelihood_per_action : [ float ]
            Average likelihood of an action considering human data form the exp.
            Computed for each cluster
        ppl_likelihood : [ float ] 
            Overall likelihod of human data under (each of) the descriptions
        ppl_opt_score : [ float ]
            Proportion of times people took an optimal action according to the
            description. Computed for each cluster
        ppl_opt_act_score : [ float ]
            Proportion of likelihood for action taken by peopl people and likelihoods
            for actions allowed by the description. Computed for each cluster
        weights : [ float ]
            1-Epsilons for the probability model 
            Epsilon * not(description) + (1-Epsilon) * description (accuracy of
            the descriptions)
        nones : int
            Number of clusters for which the interpretation was not found
        AIC : float
            Akaike Information Criterion for the model
        BIC : float
           Bayesian Information Criterion for the model
        log_marginal_likelihood : [ float ]
            Logarithm of the marginal likelihood of the description for each of 
            them; computed with the BIC approximation
    avg_scores : dict
        As above but values are averaged over all the clusters
    """
    cwd = os.getcwd()
    filename = "strategies_" + kwargs['experiment_id'] + "_" + \
               str(kwargs['num_strategies']) + "_" + str(kwargs['num_participants']) + \
               "_" + str(kwargs['num_trajs']) + kwargs['info'] + '.txt'
    
    all_mlik, all_opt, all_ppl_mlik, all_ppl_lik, all_ppl_opt = {}, {}, {}, {}, {}
    all_ppl_act, all_w = {}, {}
    num_features = 0
    num_nones = 0
    
    num_strategies = kwargs['num_strategies']
    c_weights, num_data = cluster_importance_weighting(exp_id=kwargs['experiment_id'], 
                                                       num_clust=num_strategies, 
                                                       num_p=kwargs['num_participants'],
                                                       info=kwargs['info'])
    print('SIZES: {}, {}'.format(kwargs['num_strategies'], num_data))
    lik_matrix = np.zeros((kwargs['num_strategies'], num_data)) 
    def load_all_data(exp_id, num_clust, num_part, info):
        envs = []
        action_seqs = []
        for clust_id in range(num_clust):
            e, a, _ = load_participant_data(exp_id, num_clust, num_part, clust_id, info)
            envs += e
            action_seqs += a
        return envs, action_seqs
        
    all_envs, all_seqs = load_all_data(kwargs['experiment_id'], kwargs['num_strategies'],kwargs['num_participants'],
                                       kwargs['info'])
    
    for i in range(num_strategies):

        if i in c_weights.keys():
            res = wrapper_interpret_cluster_to_procedural_description(**kwargs, 
                                                                      strategy_num=i)
            f, n, clst_mn_lik, clst_opt, ppl_mn_lik, ppl_lik, ppl_opt, eps, ppl_act, log, fp = res
            formula, pipeline = fp[0], fp[1]
            pipe = (tuple(pipeline[0][0]), pipeline[0][1])
            if log:
                save_strategies_and_evaluation(filepath=cwd+'/interprets_procedure', 
                                               filename=filename, data=log, append=i!=0)
            all_mlik[i] = clst_mn_lik
            all_opt[i] = clst_opt
            all_ppl_mlik[i] = ppl_mn_lik
            all_ppl_lik[i] = ppl_lik
            all_ppl_opt[i] = ppl_opt
            all_ppl_act[i] = ppl_act
            all_w[i] = 1-eps
            num_features += f
            num_nones += n
            
            lik_matrix = compute_log_likelihood(envs=all_envs, pipeline=pipe, people_acts=all_seqs, 
                                                formula=formula, epsilon_formula=eps, num_formula=i,
                                                lik_matrix=lik_matrix, verbose=True)
            
        else:
            print("\n\nWARNING: ")
            print("STRATEGY {} for EXPERIMENT {}: CLUSTER WAS EMPTY".format(
                                                                i,
                                                                kwargs['experiment_id']))
                                                                
    
    all_scores = {'likelihood_per_action': all_mlik, 
                  'opt_score': all_opt, 
                  'ppl_likelihood_per_action': all_ppl_mlik,
                  'ppl_likelihood': all_ppl_lik,
                  'ppl_opt_score': all_ppl_opt, 
                  'ppl_act_score': all_ppl_act,
                  'weights': all_w,
                  'nones': num_nones}
    avg_scores = [sum(score.values())/num_strategies for score in list(all_scores.values())[:-1]] +\
                 [all_scores['nones']/num_strategies]
    
    ll_data = compute_all_data_log_likelihood(weights=[1./kwargs['num_strategies']]*kwargs['num_strategies'],
                                               lik_matrix=lik_matrix)       
    ll_clust = sum(ll_data)
                 
    AIC = 2 * num_features - 2 * ll_clust
    BIC = num_features * np.log(num_data) - 2 * ll_clust
    all_scores['AIC'] = AIC
    all_scores['BIC'] = BIC
    all_scores['log_marginal_likelihood'] = -1/2 * BIC
    
    if log:
        avg = "\n\nAVERAGE OVER ALL CLUSTERS:\n\n"
        log_marginal_lik = all_scores['log_marginal_likelihood']
        numb = "ScoreLL: {}\nOPT: {}\nPPL_ScoreLL: {}\n".format(avg_scores[0],
                                                                avg_scores[1],
                                                                avg_scores[2]) + \
               "PPL_OPT: {}\nPPL_LL: {}\nPPL_ACT: {}\nW: {}\n".format(avg_scores[4],
                                                                      avg_scores[3],
                                                                      avg_scores[5],
                                                                      avg_scores[6]) + \
               "\n\nMODEL LOG MARGINAL LIKELIHOOD: {}".format(log_marginal_lik) + \
               "\nAIC: {}\nBIC: {}".format(AIC, BIC)
               
        save_strategies_and_evaluation(filepath=cwd+'/interprets_procedure', 
                                       filename=filename, data=avg+numb, append=True)
    return all_scores, avg_scores

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
    parser.add_argument('--log', '-l',
                        type=bool,
                        help="Whether to log the interpretation and their evaluations",
                        default=True)
                        
    args = parser.parse_args()
    
    interpret_evaluate_clusters(experiment_id=args.experiment_id,
                                num_strategies=args.num_strategies,
                                num_participants=args.num_participants,
                                num_demos=args.num_demos,
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
                                mean_reward=args.mean_reward,
                                expert_reward=args.expert_reward,
                                info=args.info,
                                log=args.log)
