from interpret_formula import load_EM_data, interpret_human_data
from formula_procedure_transformation import ConversionError, DNF2LTL, trajectorize
from evaluate_procedure import load_participant_data, compute_scores
from formula_visualization import prettify
from RL2DT.hyperparams import *
from hyperparams import ALLOWED_PREDS, REDUNDANT_TYPES
import argparse
import pickle
import os

class IncompleteInputError(Exception):
    """
    Exception when the argument is None.
    """
    def __init__(self, inputs):
        self.inputs = inputs
        self.message = "The input(s) {} were None and ".format(inputs) + \
                       "need to be corrected."
        super().__init__(self.message)

def load_interpreted_data(formula_filename, preds_filename, demos_filename):
    """
    Load an input DNF formula along with all the predicates and the demos used 
    in creating the formula. 

    Parameters
    ----------
    formula_filename : str
        Name of the file with the formula to recover
    preds_filename : int 
        Name of the file with the predicate matrix
    demos_filename : int
        Name of the file with the demonstrations
        
    Returns
    -------
    formula : str
        DNF formula describing a planning strategy represented by demos
    pred_matrix : tuple
        A tuple with the data used in the AI-Interpret algorithm that finds the
        DNF formula
        data_matrix : np.array(X)
            Matrix with "active" predicates for every positive/negative 
            demonstration
        X : csr_matrix
            X.shape = (num_trajs, num_preds)
        y : [ int(bool) ]
            y.shape = (num_trajs,)
            Vector indicating whether the demonstration was positive or negative
    demos : dict
        pos: [ (IHP.modified_mouselab.Trial, int) ]
            Demonstrated (state, action) pairs
        neg: [ (IHP.modified_mouselab.Trial, int) ]
            (state, action) pairs for encountered states and non-optimal actions 
            according to the RL policy
        leng_pos: [ int ]
            Lengths of the consecutive demonstrations
        leng_neg: [ int ]
            Lengths of the negative examples for the consecutive demonstrations
    """
    cwd = os.getcwd()
    
    formula_path = cwd+'/interprets_formula/'
    folder_path = cwd+'/demos/'
    
    print("Retrieving the formula...")
    with open(formula_path + formula_filename, 'rb') as handle:
        dict_object = pickle.load(handle)
    formula = list(dict_object.keys())[0]
    print('Done')
    
    print("Retrieving the predicate matrix...")
    with open(folder_path + preds_filename, 'rb') as handle:
        pred_matrix = pickle.load(handle)
    print('Done')
        
    print("Retrieving the demos...")
    folder_path = cwd+'/demos/'
    with open(folder_path + demos_filename, 'rb') as handle:
        demos = pickle.load(handle)
    print('Done')
    
    return formula, pred_matrix, demos

def wrapper_interpret_human_data(**kwargs):
    """
    Wrapper function to extract a DNF formula and its predicate matrix by either
    loading it or computing using the passed arguments. 
        
    Returns
    -------
    (See load_interpreted_data)
    formula : str
    pred_matrix : tuple
    demos : dict
    (See load_EM_data in interpret_formula)
    pipeline : [ ( [ int ], function ) ]
    weights : [ float ]
    features : [ [ float ] ]
    normalized_features : [ [ float in [0,1] ] ]
    """
    args = ['experiment_id', 'num_strategies', 'num_participants', 
            'strategy_num', 'num_trajs']
    bad_args = [a for a in args if kwargs[a] == None]
    if bad_args != []:
        raise IncompleteInputError(bad_args)
    
    res = load_EM_data(exp_num=kwargs['experiment_id'], 
                       num_strategies=kwargs['num_strategies'],
                       num_participants=kwargs['num_participants'],
                       strategy_num=kwargs['strategy_num'],
                       num_simulations=kwargs['num_trajs'])
    pipeline, weights, features, normalized_features = res
    
    end = kwargs['experiment_id'] + "_" + str(kwargs['strategy_num']) + "_" + \
          str(kwargs['num_strategies']) + "_" + str(kwargs['num_participants']) + \
          "_" + str(kwargs['num_trajs'])
    if kwargs['info'] != '':
        end += "_" + kwargs['info'] + ".pkl"
    else:
        end += ".pkl"
    formula_flnm = "human_" + end
    preds_flnm = "DSL_" + end
    demos_flnm = "human_data_" + end

    try:
        formula, pred_matrix, demos = load_interpreted_data(formula_filename=formula_flnm, 
                                                            preds_filename=preds_flnm, 
                                                            demos_filename=demos_flnm)
                                                            
    except FileNotFoundError:
        data = interpret_human_data(pipeline=pipeline, 
                                    weights=weights, 
                                    features=features,
                                    normalized_features=normalized_features,
                                    strategy_num=kwargs['strategy_num'], 
                                    all_strategies=kwargs['num_strategies'],
                                    num_demos=kwargs['num_trajs'], 
                                    exp_name=kwargs['experiment_id'], 
                                    num_participants=kwargs['num_participants'], 
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
                                    mean_rew=kwargs['mean_reward'],
                                    expert_rew=kwargs['expert_reward'],
                                    info=kwargs['info'])
        formula, pred_matrix, demos = load_interpreted_data(formula_filename=formula_flnm, 
                                                            preds_filename=preds_flnm, 
                                                            demos_filename=demos_flnm)
    return formula, pred_matrix, demos, pipeline, weights, features, normalized_features

def wrapper_compute_procedural_formula(formula, pred_matrix, demos, **kwargs):
    """
    Wrapper function to produce a procedural description out of a DNF formula.
    
    It tries to find a description by first removing (all the) predicates which 
    were defined as redundant.
    
    For example, not(is_observed) and not(is_previous_observed_max) would not 
    have a procedural description different than itself but after removing the 
    second predicate, a procedural formula could be not(is_observed) UNTIL 
    is_previous_observed_max. 
    
    Reduncant predicates are selected by hand are are hyperparameters.
    
    If this fails, the function extracts a procedural description from the initial
    formula.
        
    Returns
    -------
    LTL_formula : str
        Procedural description in readable form
    raw_LTL_formula : str
        Procedural description in callable form (with lambdas, colons, etc.)
    c : int
        Complexity (the number of used predicates) in the description
    """
    pretty_form = prettify(formula) if formula != None else formula
    print("\n\n\n\n\n" +\
          "COMPUTING PROCEDURE OUT OF {}: \n\n\n\n\n".format(pretty_form))
    trajs = trajectorize(demos)
    num_pos_demos = sum(pred_matrix[-1])
    pred_pos_matrix = pred_matrix[0][:num_pos_demos]
    
    envs, action_seqs = load_participant_data(exp_id=kwargs['experiment_id'],
                                              num_clust=kwargs['num_strategies'],
                                              num_part=kwargs['num_participants'],
                                              clust_id=kwargs['strategy_num'])
    
    try:
        LTL_formula, c, raw_LTL_formula = DNF2LTL(
                                            phi=formula, 
                                            trajs=trajs, 
                                            predicate_matrix=pred_pos_matrix,
                                            allowed_predicates=ALLOWED_PREDS,
                                            redundant_predicates=ALLOWED_PREDS,
                                            redundant_predicate_types=REDUNDANT_TYPES,
                                            p_envs=envs,
                                            p_actions=action_seqs,
                                            p_pipeline=kwargs['pipeline'])
    except ConversionError:
        LTL_formula, c, raw_LTL_formula = DNF2LTL(phi=formula, 
                                                  trajs=trajs, 
                                                  predicate_matrix=pred_pos_matrix,
                                                  allowed_predicates=ALLOWED_PREDS,
                                                  redundant_predicates=[],
                                                  p_envs=envs,
                                                  p_actions=action_seqs,
                                                  p_pipeline=kwargs['pipeline'])
    print('\n\nPROCEDURAL FORMULA:\n\n{}\n\nComplexity: {}'.format(LTL_formula, c))
    return LTL_formula, raw_LTL_formula, c


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
    parser.add_argument('--strategy_num', '-c',
                        type=int,
                        help="Number of the strategy to interpret.")
    parser.add_argument('--demo_path', '-dp',
                        type=str,
                        help="Path to the file with the demonstrations.", 
                        default='')
    parser.add_argument('--num_trajs', '-n', 
                        type=int,
                        help="How many trajs to use for interpretation.")
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
    
    res = wrapper_interpret_human_data(experiment_id=args.experiment_id,
                                       num_strategies=args.num_strategies,
                                       num_participants=args.num_participants,
                                       num_trajs=args.num_trajs,
                                       strategy_num=args.strategy_num,
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
                                             demos=demos)
    proc_formula, raw_proc_formula, c = res
    
    res = wrapper_evaluate_procedural_formula(experiment_id=args.experiment_id,
                                              num_strategies=args.num_strategies,
                                              num_participants=args.num_participants,
                                              strategy_num=args.strategy_num,
                                              raw_formula=raw_proc_formula,
                                              weights=weights,
                                              pipeline=pipeline,
                                              softmax_features=features,
                                              softmax_normalized_features=normalized_features)
