import os
import pickle
import pandas as pd
import numpy as np
from numba import jit
import numpy.linalg as LA
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.stats import gamma
from distributions import Categorical, Normal
from statsmodels.nonparametric.smoothers_lowess import lowess
import time
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
from functools import partial

small_level_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3,
                   5: 1, 6: 2, 7: 3, 8: 3, 9: 1, 10: 2, 11: 3, 12: 3}
level_values = [[0], [-4, -2, 2, 4], [-8, -4, 4, 8], [-48, -24, 24, 48]]
const_var_values = [[-10, -5, 5, 10]]

reward_levels = {'high_increasing': level_values[1:], 
                 'high_decreasing': level_values[1:][::-1],
                 'low_constant': const_var_values*3, 
                 'large_increasing': list(zip(np.zeros(5), [1,2,4,8,32]))}

reward_type = {'F1': 'categorical', 'c1.1': 'categorical', 
               'c2.1': 'categorical', 'T1.1': 'normal', 'v1.0': 'categorical'}

num_strategies = 89

eps = np.finfo(float).eps


def get_data(version, data_path='../data'): 
    data = {}
    for file in glob('{}/human/{}/*.csv'
                     .format(data_path, version)):
        name = os.path.basename(file)[:-4]
        df = pd.read_csv(file)
        parse_json(df)
        data[name] = drop_nan_cols(df)

    # n_trials = df.pid.value_counts().max()
    # complete = df.pid.value_counts(sort=False).where(lambda x:
    #                                                  x==n_trials).dropna().index
    # df = df.set_index('pid').ix[complete].reset_index()
    # pdf = pdf.ix[complete]
    
    return data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def pickle_load(file_path):
    """
        Load the pickle file localted at 'filepath'
        Params:
            file_path  -- Location of the file to be loaded.
        Returns:
            Unpickled object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    obj = pickle.load(open(file_path, "rb"))
    return obj


def create_dir(file_path):
    """
        Create directory if it does not exist
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)


implemented_features = pickle_load(f"data/implemented_features.pkl")


def pickle_save(obj, file_path):
    """
        Pickle the object
        Params:
            obj: The object to be pickled
            file_path: The path the object has to be pickled to
    """
    pickle.dump(obj, open(file_path, "wb"))


def string_to_bool(truth):
    if truth.lower() == "true":
        return True
    else:
        return False

def bool_to_string(truth):
    return "true" if truth else "false"


def get_clicks(exp_num="v1.0"):
    """
        Get clicks made by a particular participant
        Params:
            exp_num : Experiment number according to the experiment folder
            participant_num : The id of the participants to get the environments for
        Returns:
            clicks_data: The clicks made by the participant in all trials.
    """
    data = get_data(exp_num)
    mdf = data['mouselab-mdp']
    clicks_data = defaultdict(list)
    for _, row in mdf.iterrows():
        pid = row.pid
        queries = row.queries['click']['state']['target']
        queries = [int(query) for query in queries]
        queries.append(0)
        clicks_data[pid].append(queries)
    clicks_data = dict(clicks_data)
    return clicks_data


def get_participant_scores(exp_num="v1.0", num_participants=166):
    """
        Get scores of participants
        Params:
            exp_num : Experiment number according to the experiment folder.
            num_participants: Max pid+1 to consider
        Returns:
            A dictionary of scores of participants with pid as key and rewards as values.
    """
    data = get_data(exp_num)
    mdf = data['mouselab-mdp']
    participant_scores = {}
    for participant_num in range(num_participants):
        score_list = list(mdf[mdf.pid == participant_num]['score'])
        participant_scores[participant_num] = score_list
    return participant_scores


def get_environments(participant_num, exp_num="v1.0"):
    """
        Get environments of a particular participant
        Params:
            exp_num : Experiment number according to the experiment folder
            participant_num : The id of the participants to get the environments for
        Returns:
            envs: The trial values that the participant observed
    """
    data = get_data(exp_num)
    mdf = data['mouselab-mdp']
    mdf = mdf[mdf.pid == participant_num]
    envs = []
    for _, row in mdf.iterrows():
        values = row.state_rewards
        values[0] = 0
        envs.append(values)
    return envs


def get_taken_paths(participant_num, exp_num="F1"):
    data = get_data(exp_num)
    mdf = data['mouselab-mdp']
    mdf = mdf[mdf.pid == participant_num]
    taken_paths = []
    for _, row in mdf.iterrows():
        path = row.path
        taken_paths.append([int(p) for p in path])
    return taken_paths

def construct_repeated_pipeline(branching, reward_function, num_trials):
    return [(branching, reward_function)]*num_trials

def construct_pipeline(branchings, reward_distributions):
    return list(zip(branchings, reward_distributions))

def reward_function(depth, level_distributions):
    if depth > 0:
        return level_distributions[depth - 1]
    return 0.

def combine_level_dists(level_distributions):
    func = partial(reward_function, level_distributions=level_distributions)
    return func

def construct_reward_function(params_list, dist_type = 'categorical'):
    if dist_type.lower() == 'categorical':
        level_distributions = [Categorical(param) for param in params_list]
    elif dist_type.lower() == 'normal':
        level_distributions = [Normal(param[0], param[1]) for param in params_list]
    else:
        raise ValueError('Please select one of categorical or normal distibutions')
    return combine_level_dists(level_distributions)
    
def get_participant_details(pid, exp_num, get_envs=True, get_scores=True, 
                            get_clicks=True, get_taken_paths=True):
    data = get_data(exp_num)
    mdf = data['mouselab-mdp']
    mdf = mdf[mdf.pid == pid]
    scores = []
    envs = []
    taken_paths = []
    clicks_data = []
    if get_scores:
        scores = list(mdf['score'])
    for _, row in mdf.iterrows():
        if get_envs:
            values = row.state_rewards
            values[0] = 0
            envs.append(values)
        if get_taken_paths:
            path = row.path
            taken_paths.append([int(p) for p in path])
        if get_clicks:
            pid = row.pid
            queries = row.queries['click']['state']['target']
            queries = [int(query) for query in queries]
            queries.append(0)
            clicks_data.append(queries)
    return envs, scores, clicks_data, taken_paths


def get_participant_weights(participant_num, exp_num="F1", criterion="all_features"):
    try:
        if criterion == "all_features":
            participant_weights = pickle_load(
                f"data/starting_weights_{exp_num}.pkl")
        elif type(criterion) == int:
            participant_weights = pickle_load(
                f"data/starting_weights_{exp_num}_{criterion}.pkl")
        elif criterion == "normalize":
            participant_weights = pickle_load(
                f"data/starting_weights_{exp_num}_normalized.pkl")
    except FileNotFoundError:
        print("Unable to load prior weights")
        return []
    return participant_weights[participant_num]


def sidak_value(significance_threshold, num_tests):
    return 1 - (1-significance_threshold)**(1/num_tests)


def sigmoid(x):
    """
        Return the value of the sigmoid function
    """
    return 1/(1+np.exp(-x))


def temp_sigmoid(x, t):
    """
        Return sigmoid with the temperature parameter 
    """
    return sigmoid((1/t)*x)


def convert_zeros_to_none(total_trial_actions):
    """
        Make actions amenable to the Computational Microscope by replacing 
        action 0 to None
    """
    modified_actions = []
    for trial_actions in total_trial_actions:
        modified_actions.append(
            [action if action != 0 else None for action in trial_actions])
    return modified_actions


def convert_none_to_zeros(total_trial_actions):
    """
        Make actions amenable to the Computational Microscope by replacing 
        action 0 to None
    """
    modified_actions = []
    for trial_actions in total_trial_actions:
        modified_actions.append(
            [action if action != None else 0 for action in trial_actions])
    return modified_actions


def get_zero_params(params):
    return {k: 0 for k in params.keys()}


def compute_error_aic(num_parameters, error):
    """
        Compute AIC using the value of negative log likelihood
    """
    error_aic = 2*num_parameters + 2*error
    return error_aic


def compute_likelihood_aic(num_parameters, likelihood):
    """
        Compute AIC using the value of the likelihood
    """
    likelihood_aic = 2*num_parameters - 2*np.log(likelihood)
    return likelihood_aic


def get_excluded_participants(exp_num="F1", exclude_condition="MCFB"):
    conditions = {"NOFB": 0, "MCFB": 1, "ActionFB": 2}
    data = get_data(exp_num)
    pdf = data['participants']
    pdf = pdf[~(pdf.condition == conditions[exclude_condition])]
    return list(set(pdf.pid))


def get_normalized_feature_values(feature_values, features_list, max_min_values):
    """
        Get the normalized feature values
    """
    normalized_features = np.array(feature_values)
    if max_min_values:
        max_feature_values, min_feature_values = max_min_values
        for i, (feature, fv) in enumerate(zip(features_list, feature_values)):
            max_min_diff = max_feature_values[feature]-min_feature_values[feature]
            f_min_diff = fv - min_feature_values[feature]
            #print(feature, f_min_diff, max_min_diff, max_min_diff - f_min_diff)
            if max_min_diff == 0:
                normalized_features[i] = 0
            else:
                normalized_features[i] = f_min_diff/max_min_diff
    return normalized_features
    
def get_normalized_features(exp_num):
    max_feature_values = pickle_load(f"data/normalized_values/{exp_num}/max.pkl")
    min_feature_values = pickle_load(f"data/normalized_values/{exp_num}/min.pkl")
    return max_feature_values, min_feature_values

def get_transformed_weights(participant_num, trial_weights, 
                            trial_features, features_list):
    """
        Get transformed weights according the feature list across all trials by 
        zeroing all features in a trial that are not inferred
        Params:
            participant_num: ID of the participant
            trial_weights: Trial wise weights for all participants
            trial_features: Trial wise features for all participants
            features_list: The list of features according to which the weights 
                           are to be transformed
        Returns:
            weights: Transformed weights according to the feature list
    """
    t_f = features_list
    tw_f = trial_features[participant_num]
    t_w = trial_weights[participant_num]
    feature_index = {f: i+1 for i, f in enumerate(t_f)}
    weights = np.zeros((len(tw_f), len(t_f)+4))
    for i, W in enumerate(t_w):
        weights[i][0] = W[0]
    for i, F in enumerate(tw_f):
        for j, f in enumerate(F):
            weights[i][feature_index[f]] = t_w[i][j+1]
    for i, w in enumerate(t_w):
        weights[i][-3:] = w[-3:]
    return weights


def break_ties_random(a):
    """
        Get the max element by breaking ties randomly
        Params: 
            a : 1D list or ndarray
        Returns:
            random_max : Index of max element with ties broken randomly
    """
    # Takes a 1-D list and returns the index of a max element with tie broken randomly
    max_value = np.max(a)
    indices = [i for i in range(len(a)) if a[i] == max_value]
    random_max = np.random.choice(indices)
    return random_max


def get_random_small_env():
    """
        Generate a random environment of the general Mouselab-MDP
        Returns:
            env : Mouselab-MDP trial generated using the value distribution 
                  specified in level values global variable
    """
    env = []
    for index in range(0, 13):
        env.append(np.random.choice(level_values[small_level_map[index]]))
    return env


def generate_small_envs(num_envs):
    """
        Generate multiple small random environments
        Params:
            num_envs: Number of random environments to generate
        Returns:
            List of random environments
    """
    envs_list = []
    for _ in range(num_envs):
        env = get_random_small_env()
        envs_list.append(env)
    return envs_list


def get_proportion_dict(data):
    """
        Normalize values by total sum of values
        Params:
            data: dictionary with integer or float value
        Returns:
            proportions_data : normalized data dictionary
    """
    values_sum = sum(list(data.values()))
    proportions_data = {pid: value/values_sum for pid, value in data.items()}
    return proportions_data


def smoothen(freq_list, ratio=0.18):
    lis_size = len(freq_list)
    smoothed_freq_list = lowess(freq_list, range(
        lis_size), is_sorted=True, frac=ratio, it=0)
    return smoothed_freq_list[:, -1]


def bootstrapping_median_std(values, num_elements=30, num_simulations=10000):
    medians = []
    for i in range(num_simulations):
        vals = np.random.choice(values, num_elements)
        median = np.median(vals)
        medians.append(median)
    return np.std(medians)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def cint(a):
    ret = scipy.stats.t.interval(0.95, len(a)-1, 
                                 loc=np.mean(a), 
                                 scale=scipy.stats.sem(a))
    return ret


def get_squared_performance_error(participant_performance, algorithm_performance):
    """
        Compute the squared performance error between the participant and the 
        algorithm
        Params:
            participant_performance: Rewards over n trials of the participant 
                                     (1D list)
            algorithm_performance: Rewards over num_runs*n trials of the 
                                   algorithm (2D list)
        Returns:
            performance_error: Peformance error between the participant and the 
                               algorithm
    """
    algorithm_performance = np.asarray(algorithm_performance)
    num_dims = algorithm_performance.ndim
    if num_dims == 1:
        algorithm_performance = [algorithm_performance]
    participant_performance = np.asarray(participant_performance)
    mean_algorithm_performance = np.mean(algorithm_performance, axis=0)
    performance_error = np.mean(
        (participant_performance - mean_algorithm_performance)**2, axis=0)
    return performance_error


def get_squared_pe(participant_performance, algorithm_performance):
    """
        Compute the squared performance error between the participant and the 
        algorithm
        Params:
            participant_performance: Rewards over n trials of the participant 
                                    (1D list)
            algorithm_performance: Rewards over num_runs*n trials of the 
                                   algorithm (2D list)
        Returns:
            performance_error: Peformance error between the participant and the 
                               algorithm
    """
    algorithm_performance = np.asarray(algorithm_performance)
    num_dims = algorithm_performance.ndim
    if num_dims == 1:
        algorithm_performance = [algorithm_performance]
    algorithm_performance = np.array(algorithm_performance)
    participant_performance = np.asarray(participant_performance)
    participant_performance = np.expand_dims(participant_performance, axis=0)
    performance_error = np.mean(
        (participant_performance - algorithm_performance)**2)
    return performance_error


def get_absolute_performance_error(participant_performance, algorithm_performance):
    """
        Compute the absolute performance error between the participant and the 
        algorithm
        Params:
            participant_performance: Rewards over n trials of the participant 
                                    (1D list)
            algorithm_performance: Rewards over num_runs*n trials of the 
                                   algorithm (2D list)
        Returns:
            performance_error: Peformance error between the participant and the 
                               algorithm
    """
    algorithm_performance = np.asarray(algorithm_performance)
    num_dims = algorithm_performance.ndim
    if num_dims == 1:
        algorithm_performance = [algorithm_performance]
    participant_performance = np.asarray(participant_performance)
    mean_algorithm_performance = np.mean(algorithm_performance, axis=0)
    performance_error = np.mean(np.absolute(
        participant_performance - mean_algorithm_performance), axis=0)
    return performance_error


def get_performance_error(participant_performance, algorithm_performance):
    """
        Compute the performance error between the participant and the algorithm 
        (non squared)
        Params:
            participant_performance: Rewards over n trials of the participant 
                                    (1D list)
            algorithm_performance: Rewards over num_runs*n trials of the 
                                   algorithm (2D list)
        Returns:
            performance_error: Peformance error between the participant and the 
                               algorithm
    """
    algorithm_performance = np.asarray(algorithm_performance)
    num_dims = algorithm_performance.ndim
    if num_dims == 1:
        algorithm_performance = [algorithm_performance]
    participant_performance = np.asarray(participant_performance)
    mean_algorithm_performance = np.mean(algorithm_performance, axis=0)
    performance_error = np.mean(
        (participant_performance - mean_algorithm_performance), axis=0)
    return performance_error


def get_weight_distance(participant_weights, algorithm_weights):
    """
        Compute the distance in the weight space between the participant and the 
        algorithm
        Params:
            participant_weights: 2D list or 2D numpy array with weights of a 
                                 given feature set over n trials
            algorithm_weights: 3D list or 3D numpy array with weights of a given 
                               feature set over runs and trials
            The second and third dimension of both these should match.
        Returns:
            average_euclidean_distance: Average Euclidean distance between the 
                                        algorithm and the participants' weights
    """
    participant_weights = np.array(participant_weights)
    algorithm_weights = np.array(algorithm_weights)
    participant_dims = participant_weights.ndim
    algorithm_dims = algorithm_weights.ndim
    if algorithm_dims != 3:
        raise ValueError("The number of dimensions in algorithm_weights should" + \
                         " be 3. " + f"Input dimensions are {algorithm_dims}")
    elif participant_dims != 2:
        raise ValueError("The number of dimensions in participant_weights " + \
                         "should be 2. " + f"Input dimensions are {algorithm_dims}")
    participant_weight_shape = participant_weights.shape
    algorithm_weights_shape = algorithm_weights.shape
    if algorithm_weights_shape[1:] != participant_weight_shape:
        raise ValueError("The second and third dimensions of the input weights " + \
                         "do not match. " + f"{algorithm_weights_shape[1:]} " + \
                         f"and {participant_weight_shape}")
    a_beta = np.expand_dims(algorithm_weights[:, :, -3], axis=2)
    a_w = np.mean(np.multiply(algorithm_weights[:, :, :-3], a_beta), axis=0)
    p_beta = np.expand_dims(participant_weights[:, -3], axis=1)
    p_w = np.multiply(participant_weights[:, :-3], p_beta)
    num_runs = a_w.shape[0]
    num_trials = a_w.shape[1]
    average_euclidean_distance = np.squeeze(
        np.sum(np.sqrt(np.sum((a_w-p_w)**2, axis=1)), axis=None))/num_trials
    return average_euclidean_distance


def normalize(x):
    normalized_x = x / LA.norm(x)
    return normalized_x


def normalize_weights(weights):
    weights = weights.tolist()
    for i in range(len(weights)):
        temp = normalize(weights[i])
        weights[i] = temp
    return np.array(weights)


def get_normalized_weight_distance(participant_weights, algorithm_weights):
    """
        Compute the normalized distance in the weight space between the 
        participant and the algorithm
        Params:
            participant_weights: 2D list or 2D numpy array with weights of a 
                                 given feature set over n trials
            algorithm_weights: 3D list or 3D numpy array with weights of a given 
                               feature set over runs and trials
            The second and third dimension of both these should match.
        Returns:
            average_euclidean_distance: Average Euclidean distance between the 
                                        algorithm and the participants' weights
    """
    participant_weights = np.array(participant_weights)
    algorithm_weights = np.array(algorithm_weights)
    participant_dims = participant_weights.ndim
    algorithm_dims = algorithm_weights.ndim
    if algorithm_dims != 3:
        raise ValueError("The number of dimensions in algorithm_weights should" + \
                         " be 3. " + f"Input dimensions are {algorithm_dims}")
    elif participant_dims != 2:
        raise ValueError("The number of dimensions in participant_weights " + \
                         "should be 2. " + f"Input dimensions are {algorithm_dims}")
    participant_weight_shape = participant_weights.shape
    algorithm_weights_shape = algorithm_weights.shape
    if algorithm_weights_shape[1:] != participant_weight_shape:
        raise ValueError("The second and third dimensions of the input weights " + \
                         "do not match. " + f"{algorithm_weights_shape[1:]} " + \
                         f"and {participant_weight_shape}")
    a_beta = np.expand_dims(algorithm_weights[:, :, -3], axis=2)
    a_w = np.mean(np.multiply(algorithm_weights[:, :, :-3], a_beta), axis=0)
    p_beta = np.expand_dims(participant_weights[:, -3], axis=1)
    p_w = np.multiply(participant_weights[:, :-3], p_beta)
    num_runs = a_w.shape[0]
    num_trials = a_w.shape[1]
    a_w = normalize_weights(a_w)
    p_w = normalize_weights(p_w)
    average_euclidean_distance = np.squeeze(
        np.sum(np.sqrt(np.sum((a_w-p_w)**2, axis=1)), axis=None))/num_trials
    return average_euclidean_distance


def compute_rpe(r, init_pe=0):
    """
        Compute the reward prediction error of a given sequence of rewards
        Params:
            r: Rewards across trials (1D list or ndarray)
            init_prediction: The initial prediction error
        Returns:
            rpe: Reward prediction Error
    """
    r = np.asarray(r)
    num_trials = r.shape[0]
    cumulative_average_score_list = [init_pe]
    cumulative_score_list = np.cumsum(r)
    for i in range(1, num_trials):
        cumulative_average_score_list.append(cumulative_score_list[i-1]/i)
    rpe = [np.absolute(r[i] - cumulative_average_score_list[i])
           for i in range(num_trials)][:num_trials-1]
    return rpe


def compute_weight_changes(W):
    """
        Compute the norm of weight difference across trials
        Params:
            W: The weight matrix of shape (num_trials, num_features)
        Returns:
            norms: Euclidean norm of weight changes across trials. (Size is 
                   (num_trials - 1))
    """
    W = np.asarray(W)
    norms = []
    num_trials = W.shape[0]
    for i in range(1, num_trials):
        w_diff = W[i] - W[i-1]
        norm = LA.norm(w_diff, 2)
        norms.append(norm)
    norms = np.asarray(norms)
    return norms


def compute_average_transition_matrix(S):
    """
        Compute the average transition for all participants combined
        Params:
            S: Inferred strategy sequences for num_participants * num_runs * n_trials.
               (Should be a 3D list or 3D array)
        Returns:
            average_transition_matrix: The computed average transition matrix
    """
    S = np.asarray(S)
    S_dims = S.ndim
    if S_dims != 3:
        raise ValueError(
            f"Number of dimensions of 'S' should be 3 given {S_dims}")

    num_participants = S.shape[0]
    num_runs = S.shape[1]
    num_trials = S.shape[2]
    transition_matrix = np.zeros((38, 38))
    for i in range(num_participants):
        for j in range(num_runs):
            for k in range(num_trials-1):
                transition_matrix[S[i][j][k]][S[i][j][k+1]] += 1
    out = np.zeros_like(transition_matrix)
    row_sum = transition_matrix.sum(axis=1, keepdims=True)
    average_transition_matrix = np.divide(
        transition_matrix, row_sum, out=out, where=row_sum != 0)
    return average_transition_matrix


def get_strategy_counts(S):
    """
        Get the strategy count at each time step
        Params:
            S: Inferred strategy sequences for n_participants * num_runs * n_trials. 
               (Should be a 3D list or 3D array)
        Returns:
            strategy_count : ndarray of dimension n_strategies * n_trials
    """
    S = np.asarray(S)
    S_dims = S.ndim
    if S_dims != 3:
        raise ValueError(
            f"Number of dimensions of 'S' should be 3 given {S_dims}")

    num_participants = S.shape[0]
    num_runs = S.shape[1]
    num_trials = S.shape[2]
    strategy_count = np.zeros((num_trials, num_strategies))
    for i in range(num_participants):
        for j in range(num_runs):
            for k in range(num_trials):
                strategy_count[k][S[i][j][k]] += 1
    strategy_count = strategy_count / (num_runs*num_participants)
    return strategy_count


def get_delay_penalty(q_data, env, action_sequence):
    """
        Get delay penalty of an environment
        Params:
            q_data: Pickle file present in the results folder
            env: Environment represented as node values from 0 to 12
            action_seqnece: List of actions to get delays for (ends with 0)
        Returns:
            delays: Delay penalties for all actions

    """
    env_q = q_data[tuple(env)]
    delays = []
    env_copy = env.copy()
    env_copy = ['_']*13
    env_copy[0] = '0'
    for action in action_sequence:
        if action == 0:
            action = 13
        s_rep = " ".join(env_copy)
        action_q_data = env_q[s_rep]
        max_action_q = max(action_q_data.values())
        present_action_q = action_q_data[str(action)]
        if present_action_q == max_action_q:
            delay = 0
        else:
            delay = 2 + max_action_q - present_action_q
        delays.append(delay)
        if action is not 13:
            env_copy[action] = str(env[action])
    return delays


def cholesky_decomposition(A):
    return scipy.linalg.cholesky(A, lower=False)


def sample_mvnrnd_precision(mean, precision):
    n_dim = precision.shape[0]
    L = cholesky_decomposition(precision)
    Z = np.random.randn(n_dim)
    res = LA.solve(L, Z)
    sample = res + mean
    return sample


def sample_gaussian_precision(mean, precision):
    covariance = LA.inv(precision)
    return np.random.multivariate_normal(mean, covariance)


def estimate_bayes_glm(X, y, prior_mean, prior_precision, a, b):
    X = np.asarray(X, dtype=np.float64)
    X_new = np.expand_dims(X, axis=0)
    H = X_new.T.dot(X_new) + prior_precision
    prior_mean = np.expand_dims(prior_mean, axis=1)
    res = prior_precision.dot(prior_mean)
    res2 = X.T*y
    res2 = np.expand_dims(res2, axis=1)
    mu = np.linalg.solve(H, res2 + res).reshape(-1)
    n = 1
    a = a + n/2
    m_old = prior_mean.T.dot(prior_precision).dot(prior_mean)
    m_new = mu.T.dot(H).dot(mu)
    b = b + 0.5*(y**2 + m_old - m_new)
    return mu, H, a, b


def sample_coeffs(prior_mean, prior_precision, a, b, n_samples=1):
    """
        TODO:
    """
    gamma_rvs = gamma.rvs(a*np.ones(n_samples), scale=(1/b)*np.ones(n_samples))
    k = np.maximum(gamma_rvs, eps)
    k = np.reshape(k, (-1, 1))
    samples = []
    for i in range(n_samples):
        new_p = k[i]*prior_precision
        sample = sample_mvnrnd_precision(prior_mean, new_p)
        samples.append(sample)
    samples = np.array(samples)
    return samples


def mse(participant_learning_curve, model_learning_curve):
    """
        Compute MSE for the arguments
        Returns:
            Sum of MSE between the arguments
    """
    model_learning_curve = np.array(model_learning_curve)
    participant_learning_curve = np.array(participant_learning_curve)
    return np.sum((model_learning_curve - participant_learning_curve)**2)


def total_participant_mse(participants_learning_curves, model_participants_curves):
    """
        Compute sum of MSEs for individual participants
        The arguments should only contain pids for which the sum is to be 
        calculated as keys
    """
    total_mse = 0.0
    for pid in participants_learning_curves.keys():
        model_participant_curve = rows_mean(model_participants_curves[pid])
        total_mse += mse(participants_learning_curves[pid],
                         model_participant_curve)
    return total_mse


def clicks_overlap(participant_clicks, algorithm_clicks):
    """
        Get the average value of Ratio of A ^ B / A U B for the participant and 
        algorithm clicks
        Params:
            Participant_clicks, algorithm_clicks : Dictionary of clicks made by
                                                   participants, algorithm 
                                                   respectively.(Pids are keys)
            Assumes that algorithm_clicks consist of multiple simulations.
        Returns:
            Average value of the ratio across trials and across simulations for 
            each participant.
    """
    def compute_average_trial_proportion(participant_click_sequence, algorithm_click_sequence):
        ratios = []
        for p_clicks, a_clicks in zip(participant_click_sequence, algorithm_click_sequence):
            p_clicks = [click for click in p_clicks if click not in [0, None]]
            a_clicks = [click for click in a_clicks if click not in [0, None]]
            p_set = set(p_clicks)
            a_set = set(a_clicks)
            intersection = p_set.intersection(a_set)
            union = p_set.union(a_set)
            if len(union) == 0:
                ratios.append(1)
            else:
                ratios.append(len(intersection)/len(union))
        return np.mean(ratios)

    participant_clicks_overlap = {}
    for pid in participant_clicks.keys():
        participant_click_sequence = participant_clicks[pid]
        algo_simulation_clicks = algorithm_clicks[pid]
        mean_ratios = []
        for algo_click_sequence in algo_simulation_clicks:
            mean_ratios.append(compute_average_trial_proportion(
                participant_click_sequence, algo_click_sequence))
        participant_clicks_overlap[pid] = np.mean(mean_ratios)
    return participant_clicks_overlap


def absolute_chosen_path_agreement(participants_chosen_paths, algorithm_chosen_paths):
    """
        Returns the agreement between the paths taken without considering the 
        other paths that were equally likely
        Params:
            participants_chosen_paths, algorithm_chosen_paths: Dictionary of 
                                                               paths taken by the
                                                               participant and the
                                                               algorithm
        Returns:
            The average agreement score between the participants and the algorithms
    """
    path_nums = {(0, 1, 2, 3): 0, (0, 1, 2, 4): 1, (0, 5, 6, 7): 2,
                 (0, 5, 6, 8): 3, (0, 9, 10, 11): 4, (0, 9, 10, 12): 5}

    def get_path_num(taken_path):
        path = tuple(taken_path)
        return path_nums[path]

    def compute_average_match(participant_chosen_paths, algorithm_chosen_paths):
        participant_paths = np.array(
            [get_path_num(trial_path) for trial_path in participant_chosen_paths])
        algorithm_paths = np.array([get_path_num(trial_path)
                                    for trial_path in algorithm_chosen_paths])
        matches = np.count_nonzero(participant_paths == algorithm_paths)
        return matches/len(participant_paths)

    participant_path_agreement = {}
    for pid in participants_chosen_paths.keys():
        participant_chosen_paths = participants_chosen_paths[pid]
        algo_chosen_paths = algorithm_chosen_paths[pid]
        averages = []
        for algorithm_chosen_path in algo_chosen_paths:
            averages.append(compute_average_match(
                participant_chosen_paths, algorithm_chosen_path))
        participant_path_agreement[pid] = np.mean(averages)
    return participant_path_agreement


def strategy_accuracy(participants_strategy_sequences, algorithm_strategy_sequences):
    """
        Returns the accuracy between the strategy's sequences and the 
        algorithm's sequences
        Params:
            participants_strategy_sequences, algorithm_strategy_sequences: 
                                                     dictionary with pid as keys
        Returns:
            Participant_wise_accuracy
    """
    def compute_participant_strategy_accuracy(participant_strategy_sequence, algorithm_strategy_sequence):
        num_trials = len(participant_strategy_sequence)
        participant_strategy_sequence = np.array(participant_strategy_sequence)
        algorithm_strategy_sequence = np.array(algorithm_strategy_sequence)
        match = np.count_nonzero(
            participant_strategy_sequence == algorithm_strategy_sequence)
        return match/participant_strategy_sequence.shape[0]

    participant_strategy_accuracy = {}
    for pid in participants_strategy_sequences.keys():
        participant_strategy_sequence = participants_strategy_sequences[pid]
        algo_strategy_sequences = algorithm_strategy_sequences[pid]
        accuracies = []
        for algorithm_strategy_sequence in algo_strategy_sequences:
            accuracies.append(compute_participant_strategy_accuracy(
                participant_strategy_sequence, algorithm_strategy_sequence))
        participant_strategy_accuracy[pid] = np.mean(accuracies)
    return participant_strategy_accuracy


def compute_transition_distance(participants_strategies, algorithm_strategies):
    """
        Get the MSE between algorithm transition matrix and participant 
        transition matrix
        participants_strategies and algorithm_strategies should be a 3D list or 
        ndarray
    """
    participant_transition_matrix = compute_average_transition_matrix(
        participants_strategies)
    algorithm_transition_matrix = compute_average_transition_matrix(
        algorithm_strategies)
    transition_distance = (participant_transition_matrix -
                           algorithm_transition_matrix)**2
    mse_distance = np.sum(transition_distance)
    return mse_distance


def make_bar_plot(x, y, figure_size=(15, 7), title='', xlabel='', ylabel='', 
                  line_label='', width=1.5, title_size=26, axes_font_size=24,
                  ticks_font_size=20, legend_size=20, tick_options={}, 
                  dir_path=None, show=True):
    """
        Makes bar plot with given inputs
        Params:
            x: X-axis of the points
            y: Y-axis of the points
            participant_num: Participant number
            algo: Name of the algorithm
            linewidth : Width of the plotting line
            title_size : Size of the title
            axes_font_size : Size of labels of the axes
            ticks_font_size : Size of labels of ticks
            legend_size : Size of the legend
            tick_options: Additional tick options
            dir_path: Path where the file should be saved
    """
    mpl.rc("savefig", dpi=150)
    x = np.asarray(x)
    y = np.asarray(y)
    plt.figure(figsize=figure_size)
    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=axes_font_size)
    plt.ylabel(ylabel, fontsize=axes_font_size)
    plt.bar(x, y, width=width)
    plt.tick_params(axis='both', labelsize=ticks_font_size)
    if 'x' in tick_options:
        plt.xticks(**tick_options['x'])
    elif 'y' in tick_options:
        plt.yticks(**tick_options['y'])
    if len(line_label) != 0:
        plt.legend(fontsize=legend_size)
    if dir_path is not None:
        plt.savefig(dir_path)
    if show:
        plt.show()
    else:
        plt.close()


def make_plot(x, y, figure_size=(15, 7), title='', xlabel='', ylabel='', 
              line_label='', width=1.5, title_size=26, axes_font_size=24,
              ticks_font_size=20, legend_size=20, tick_options={}, dir_path=None,
              show=True):
    """
        Makes plot with given inputs
        Params:
            x: X-axis of the points
            y: Y-axis of the points
            participant_num: Participant number
            algo: Name of the algorithm
            linewidth : Width of the plotting line
            title_size : Size of the title
            axes_font_size : Size of labels of the axes
            ticks_font_size : Size of labels of ticks
            legend_size : Size of the legend
            tick_options: additional tick_options
            dir_path: Path where the file should be saved
    """
    mpl.rc("savefig", dpi=150)
    x = np.asarray(x)
    y = np.asarray(y)
    plt.figure(figsize=figure_size)
    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=axes_font_size)
    plt.ylabel(ylabel, fontsize=axes_font_size)
    plt.tick_params(axis='both', labelsize=ticks_font_size)
    if 'x' in tick_options:
        plt.xticks(**tick_options['x'])
    elif 'y' in tick_options:
        plt.yticks(**tick_options['y'])
    plt.plot(x, y, linewidth=width)
    if len(line_label) != 0:
        plt.legend(fontsize=legend_size)
    if dir_path is not None:
        plt.savefig(dir_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_multiple(data={}, figure_size=(15, 7), title='', xlabel='', ylabel='', 
                  width=1.5, title_size=26, axes_font_size=24, ticks_font_size=20,
                  legend_size=20, tick_options={}, dir_path=None, show=True):
    """
        Makes plot with multiple lines
        Params:
            data : dictionary of data points (x,y) with keys as labels of lines
            participant_num: Participant number
            algo: Name of the algorithm
            linewidth : Width of the plotting line
            title_size : Size of the title
            axes_font_size : Size of labels of the axes
            ticks_font_size : Size of labels of ticks
            legend_size : Size of the legend
            tick_options: additional tick_options
            dir_path: Path where the file should be saved
    """
    mpl.rc("savefig", dpi=150)
    lines = data.keys()
    for line in lines:
        data[line]['x'] = np.asarray(data[line]['x'])
        data[line]['y'] = np.asarray(data[line]['y'])
    plt.figure(figsize=figure_size)
    plt.title(title, fontsize=title_size)
    plt.xlabel(xlabel, fontsize=axes_font_size)
    plt.ylabel(ylabel, fontsize=axes_font_size)
    plt.tick_params(axis='both', labelsize=ticks_font_size)
    if 'x' in tick_options:
        plt.xticks(**tick_options['x'])
    elif 'y' in tick_options:
        plt.yticks(**tick_options['y'])
    for line in lines:
        plt.plot(data[line]['x'], data[line]['y'], label=line, linewidth=width)
    plt.legend(fontsize=legend_size)
    if dir_path is not None:
        plt.savefig(dir_path)
    if show:
        plt.show()
    else:
        plt.close()


def plot_performance(participant_performance, algorithm_performance, 
                     participant_num=None, algo='Algorithm',width=1.5, 
                     title_size=26, axes_font_size=24, ticks_font_size=20, 
                     legend_size=20, dir_path=None, show=True):
    """
        Plot the performance of algorithm and participant
        Params:
            participant_performance: 1D or 2D list or ndarray. If 1D, it is 
                                     interpreted as a single participant's 
                                     performance and there should be participant 
                                     number. If 2D, it is interpreted as 
                                     performance of all participants.
            algorithm_performance: 1D or 2D list or ndarray. If 1D, intepreted 
                                   as mean of algorithm's performance. Else 
                                   interpreted as performance over multiple runs.
            participant_num: Participant number
            algo: Name of the algorithm
            linewidth : Width of the plotting line
            title_size : Size of the title
            axes_font_size : Size of labels of the axes
            ticks_font_size : Size of labels of ticks
            legend_size : Size of the legend
    """
    participant_performance = np.asarray(participant_performance)
    algorithm_performance = np.asarray(algorithm_performance)
    alg_dims = algorithm_performance.ndim
    participant_dims = participant_performance.ndim
    if alg_dims == 2:
        algorithm_performance = np.mean(algorithm_performance, axis=0)
    if participant_dims == 2:
        participant_performance = np.mean(participant_performance, axis=0)
    elif participant_num == None:
        raise ValueError("Missing participant number")
    num_participant_trials = participant_performance.shape[0]
    num_algorithm_trials = algorithm_performance.shape[0]
    if participant_dims == 1:
        par_label = f'Participant {participant_num}'
    else:
        par_label = 'Participants'

    mpl.rc("savefig", dpi=150)
    plt.figure(figsize=(15, 7))
    plt.title("Performance Curve", fontsize=title_size)
    plt.xlabel("Trial Number", fontsize=axes_font_size)
    plt.ylabel("Performance", fontsize=axes_font_size)
    plt.tick_params(axis='both', labelsize=ticks_font_size)
    plt.plot(range(1, num_participant_trials+1),
             participant_performance, label=par_label, linewidth=width)
    plt.plot(range(1, num_algorithm_trials+1),
             algorithm_performance, label=algo, linewidth=width)
    plt.legend(fontsize=legend_size)
    if dir_path is not None:
        plt.savefig(dir_path)
    if show:
        plt.show()
    else:
        plt.close()

def annotated_scatter_plot(x, y, names, color=None, s=100, fs=10):
    c_size = len(x)
    if color:
        c = np.random.randint(1,5,size=c_size)
    else:
        c = [1]*c_size
    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn

    fig,ax = plt.subplots()
    sc = plt.scatter(x,y,c=c, s=s, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):

        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = " ".join([names[n] + "\n" for n in ind["ind"]])
        annot.set_text(text)
        annot.set_fontsize(fs)
        annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)


    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    return fig

def columns_mean(a):
    """
        Find mean by collapsing second dimension
        Params:
            a : nD numpy array where n > 1
        Returns:
            mean: Mean by collapsing second dimension
    """
    mean = np.mean(a, axis=1)
    return mean


def rows_mean(a):
    """
        Find mean by collapsing first dimension
        Params:
            a : nD numpy array or list where n > 1
        Returns:
            mean: Mean by collapsing first dimension
    """
    mean = np.mean(a, axis=0)
    return mean


def remove_elements_at_indices(lis, exclude_indices):
    """
        Remove elements at indices
    """
    if exclude_indices:
        if type(lis) != list:
            if type(lis) == np.ndarray:
                lis = np.delete(lis, exclude_indices, axis=0)
                return lis
            else:
                lis = lis.tolist()
        lis_size = len(lis)
        lis = [lis[i] for i in range(lis_size) if i not in exclude_indices]
    return lis


def get_strategy_sequences(env, trials_data):
    """
    Get the strategy sequences of clicks made by an algorithm.

    Arguments:
        env {Gym env} -- Mouselab Gym Environment
        trials_data {dict} -- Contains details about simulated trials of
                              an algorithm (e.g actions)

    Returns:
        dict -- trials_data dict with the inferred strategy sequences added.
    """
    modified_actions = convert_zeros_to_none(trials_data['a'])
    cm = ComputationalMicroscope(
        38, weight_MAP, "small", distances_path=None, prior_path="data/gradual_transitions.pkl")
    strategy_info = cm.infer_sequences(
        modified_actions, env.get_ground_truth())
    trials_data['s'] = strategy_info[0]
    return trials_data


def get_normalized_strategy_weights():
    num_strategies = 38
    s_weights = np.zeros((38, 59))
    for s in range(num_strategies):
        s_weights[s] = pickle_load(f"data/strategy_weights/{s}.pkl")
    return s_weights

def get_counts(strategies, num_trials):
    new_strategies_list = list(strategies.values())
    new_strategies_list = [S for S in new_strategies_list if len(S) == num_trials]
    strategies_data = np.array(new_strategies_list)
    strategies_data = strategies_data.flatten()
    counts = Counter(strategies_data)
    ns = strategies_data.shape[0]
    counts = {k: v/ns for k,v in counts.items()}
    return counts

def get_modified_weights(strategy_space, weights):
    num_strategies = len(strategy_space)
    W = np.zeros((num_strategies, weights.shape[1]))
    for i, s in enumerate(strategy_space):
        W[i] = weights[s-1]
    return W

def make_clusters(D, method ='ward', cutoff=None, max_clusters = None):
    condensed_vals = squareform(D)
    Z = linkage(condensed_vals, method)
    if max_clusters :
        hierarchical_clusters = fcluster(Z, max_clusters, 'maxclust')
    else:
        hierarchical_clusters = fcluster(Z, cutoff, criterion='distance')
    return Z, hierarchical_clusters

def plot_clusters(Z, labels, scale = 'log'):
    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(111)
    if scale:
        plt.yscale(scale)
    _ = dendrogram(Z, labels=labels, ax=ax)

def get_cluster_dict(clusters, strategy_space):
    cluster_dict = defaultdict(list)
    for (c, s) in zip(clusters, strategy_space):
        cluster_dict[c].append(s)
    return dict(cluster_dict)

class Participant():
    """ 
    Creates a participant object which contains all details about the participant

    Returns:
        Participant -- Contains details such as envs, scores, clicks, taken paths,
                       strategies and weights at each trial.
    """

    def __init__(self, exp_num, pid, excluded_trials=None, get_strategies = True):
        self.exp_num = exp_num
        self.pid = pid
        print(self.pid)
        self.get_weights = False
        self.excluded_trials = excluded_trials
        res = get_participant_details(pid=self.pid, exp_num=self.exp_num)
        self.envs, self.scores, self.clicks, self.taken_paths = res
        num_excluded = len(excluded_trials) if excluded_trials else 0
        self.num_trials = len(self.clicks) - num_excluded
        self.get_strategies = get_strategies
        if self.get_strategies:
            try:
                self.strategies = pickle_load("results/final_strategy_inferences" + \
                                              f"/{self.exp_num}_strategies.pkl")
                self.strategies = np.array(self.strategies[self.pid])
                self.temperature = pickle_load("results/final_strategy_inferences" + \
                                               f"/{self.exp_num}_temperatures.pkl")
                self.temperature = self.temperature[self.pid]
            except FileNotFoundError:
                print("Inferred strategy sequence not found")
                self.strategies = np.array([1]) # Fix this
                self.temperature=1
        else:
            self.strategies = [None]*len(self.envs)       
        self.modify_included_data()
        self.first_trial_data = self.get_first_trial_data()
        self.all_trials_data = self.get_all_trials_data()

    def include_all_trials(self):
        self.__init__(self.exp_num, self.pid, excluded_trials=None)

    def modify_included_data(self):
        self.envs = remove_elements_at_indices(self.envs, self.excluded_trials)
        self.scores = remove_elements_at_indices(
            self.scores, self.excluded_trials)
        self.clicks = remove_elements_at_indices(
            self.clicks, self.excluded_trials)
        if self.get_weights:
            if np.array(self.weights).any():
                self.weights = remove_elements_at_indices(
                    self.weights, self.excluded_trials)
        if self.get_strategies:
            if self.strategies.any():
                self.strategies = remove_elements_at_indices(
                    self.strategies, self.excluded_trials)
        self.taken_paths = remove_elements_at_indices(
            self.taken_paths, self.excluded_trials)

    def get_first_trial_data(self):
        first_trial_actions = self.clicks[0]
        first_trial_rewards = [-1]*(len(first_trial_actions)-1) + \
            [self.scores[0] + len(first_trial_actions) - 1]
        first_trial_data = {'actions': first_trial_actions, 
                            'rewards': first_trial_rewards,
                            'taken_path': self.taken_paths[0], 
                            'strategy': self.strategies[0]}
        return first_trial_data

    def get_all_trials_data(self):
        actions_data = self.clicks
        rewards_data = [[-1]*(len(first_trial_actions)-1) + [self.scores[trial_num] + \
                        len(first_trial_actions) - 1] 
                        for trial_num, first_trial_actions in enumerate(actions_data)]
        total_data = {'actions': actions_data, 
                      'rewards': rewards_data,
                      'taken_paths': self.taken_paths, 
                      'strategies': self.strategies,
                      'temperature': self.temperature}
        return total_data
