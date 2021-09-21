import numpy as np
import operator
from scipy.special import softmax, logsumexp
from IHP.learning_utils import get_normalized_feature_values, get_counts
from IHP.modified_mouselab import TrialSequence
from IHP.generic_mouselab import GenericMouselabEnv
from IHP.planning_strategies import strategy_dict

def get_accuracy_position(position, ground_truth, clicks, pipeline, features, normalized_features, W):
    num_features = len(features)
    num_trials = len(ground_truth)
    env = TrialSequence(1, pipeline, ground_truth = [ground_truth])
    trial = env.trial_sequence[0]
    beta = 1
    acc = []
    total_neg_click_likelihood = 0
    for click in clicks:
        unobserved_nodes = trial.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        click_index = unobserved_node_labels.index(click)
        feature_values = np.zeros((len(unobserved_nodes), num_features))
        for i, node in enumerate(unobserved_nodes):
            feature_values[i] = node.compute_termination_feature_values(features)
            if normalized_features:
                feature_values[i] = get_normalized_feature_values(
                feature_values[i], features, normalized_features)
        dot_product = beta*np.dot(W, feature_values.T)
        softmax_dot = softmax(dot_product)
        neg_log_likelihood = -np.log(softmax_dot[click_index])
        total_neg_click_likelihood += neg_log_likelihood
        sorted_indices = np.argsort(dot_product)[::-1]
        sorted_list_indices = get_sorted_list_indices(sorted_indices, dot_product)
        sorted_list_clicks = [[unobserved_node_labels[index] for index in indices] for indices in sorted_list_indices]
        click_present = False
        for clicks_list in sorted_list_clicks[:position]:
            if click in clicks_list:
                click_present = True
                break
        if click_present:
            acc.append(1)
        else:
            acc.append(0)
        trial.node_map[click].observe()
    average_click_likelihood = np.exp((-1/len(clicks))*total_neg_click_likelihood)
    return acc, average_click_likelihood

def get_acls(strategies, pids, p_envs, p_clicks, pipeline, features, normalized_features, strategy_weights):
    acls = []
    random_acls = []
    total_acc = []
    for pid in pids:
        if pid in p_envs:
            #print(pid)
            envs = p_envs[pid]
            clicks = p_clicks[pid]
            pid_acc = []
            for i in range(len(envs)):
                strategy_accs = []
                strategy_num = strategies[pid][i]
                pid_acc, acl = get_accuracy_position(1, envs[i], clicks[i], pipeline, features, normalized_features, strategy_weights[strategy_num - 1])
                _, random_acl = get_accuracy_position(1, envs[i], clicks[i], pipeline, features, normalized_features, np.zeros(len(features)))
                acls.append(acl)
                random_acls.append(random_acl)
            total_acc += pid_acc
    #print(np.sum(total_acc)/len(total_acc))
    return acls, random_acls

def compute_average_click_likelihoods(strategies, pids, p_envs, p_clicks, pipeline, features, normalized_features, strategy_weights):
    acls = []
    random_acls = []
    for pid in pids:
        if pid in p_envs:
            print(pid)
            envs = p_envs[pid]
            clicks = p_clicks[pid]
            for i in range(len(envs)):
                strategy_accs = []
                try:
                    strategy_num = strategies[pid][i]
                    trial = TrialSequence(1, pipeline, ground_truth = [envs[i]]).trial_sequence[0]
                    num_clicks = len(clicks[i])
                    acl = (1/num_clicks)*np.exp(compute_log_likelihood(trial, clicks[i], features, strategy_weights[strategy_num-1], normalized_features))
                    random_acl = (1/num_clicks)*np.exp(compute_log_likelihood(trial, clicks[i], features, strategy_weights[38], normalized_features))
                    acls.append(acl)
                    random_acls.append(random_acl)
                except Exception as e:
                    print(e)
    return acls, random_acls

def summarize_acl(strategies, acls, random_acls, num_trials):
    counts = get_counts(strategies, num_trials)
    #print(sorted(acls))
    print("Median of average click likelihoods is ", np.median(acls))
    print("Median of random average click likelihoods is ", np.median(random_acls))
    print("Mean of average click likelihoods is ", np.mean(acls))
    print("Mean of random average click likelihoods is ", np.mean(random_acls))
    sorted_counts = sorted(counts.items(), key = operator.itemgetter(1), reverse=True)
    print(sorted_counts)

def get_sorted_list_indices(sorted_indices, dot_product):
    total_list = []
    temp_list = [sorted_indices[0]]
    for index in sorted_indices[1:]:
        dp = dot_product[index]
        if not temp_list or dp == dot_product[temp_list[-1]]:
            temp_list.append(index)
        else:
            total_list.append(temp_list)
            temp_list = []
    return total_list
    
def compute_log_likelihood(trial, click_sequence, features, weights, inv_t = False, normalized_features = False):
    trial.reset_observations()
    log_likelihoods = []
    feature_len = weights.shape[0]
    beta = 1
    W = weights
    if inv_t:
        feature_len -= 1
        beta = weights[-1]
        W = weights[:-1]
    ws = []
    fs = []
    for w, f in zip(W, features):
        if w!=0:
            ws.append(w)
            fs.append(f)
    for click in click_sequence:
        unobserved_nodes = trial.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        feature_values = trial.get_node_feature_values(unobserved_nodes, fs, normalized_features)
        dot_product = beta*np.dot(ws, feature_values.T)
        click_index = unobserved_node_labels.index(click)
        trial.node_map[click].observe()
        log_lik = dot_product[click_index] - logsumexp(dot_product)
        log_likelihoods.append(log_lik)
    return np.sum(log_likelihoods)

def compute_trial_feature_log_likelihood(trial, trial_features, click_sequence, weights, inv_t = False):
    trial.reset_observations()
    log_likelihoods = []
    feature_len = weights.shape[0]
    beta = 1
    W = weights
    if inv_t:
        feature_len -= 1
        beta = weights[-1]
        W = weights[:-1]
    for i in range(len(click_sequence)):
        click = click_sequence[i]
        unobserved_nodes = trial.get_unobserved_nodes()
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        feature_values = trial_features[i][unobserved_node_labels, :]
        dot_product = beta*np.dot(W, feature_values.T)
        click_index = unobserved_node_labels.index(click)
        trial.node_map[click].observe()
        log_lik = dot_product[click_index] - logsumexp(dot_product)
        log_likelihoods.append(log_lik)
    return np.sum(log_likelihoods)

def get_clicks(trial, features, weights, normalized_features, inv_t = False):
    trial.reset_observations()
    actions = []
    feature_len = weights.shape[0]
    beta = 1
    W = weights
    if inv_t:
        feature_len -= 1
        beta = weights[-1]
        W = weights[:-1]
    unobserved_nodes = trial.get_unobserved_nodes()
    click = -1
    while(click != 0):
        unobserved_node_labels = [node.label for node in unobserved_nodes]
        feature_values = trial.get_node_feature_values(unobserved_nodes, features, normalized_features)
        dot_product = beta*np.dot(W, feature_values.T)
        softmax_dot = softmax(dot_product)
        click = np.random.choice(unobserved_node_labels, p = softmax_dot)
        actions.append(click)
        trial.node_map[click].observe()
        unobserved_nodes = trial.get_unobserved_nodes()
    return actions

def generate_clicks(pipeline, num_trials, weights, features, normalized_features, envs=None):
    trials = TrialSequence(num_trials, pipeline, ground_truth=envs)
    clicks = []
    for trial in trials.trial_sequence:
        clicks.append(get_clicks(trial, features, weights, normalized_features))
    return trials.ground_truth, clicks

def generate_algorithm_data(strategy_num,
                        pipeline,
                        num_simulations = 1000,
                        envs=None):
    env = GenericMouselabEnv(
                        num_simulations,
                        pipeline,
                        ground_truth=envs
                    )
    simulated_actions = []
    for sim_num in range(num_simulations):
        trial = env.present_trial
        actions = strategy_dict[strategy_num](trial)
        simulated_actions.append(actions)
        env.reset_trial()
        env.get_next_trial()
    return env.ground_truth, simulated_actions

def compute_trial_features(
                    pipeline,
                    ground_truth,
                    trial_actions,
                    features_list,
                    normalized_features):
    num_features = len(features_list)
    env = TrialSequence(
                    num_trials = 1,
                    pipeline = pipeline,
                    ground_truth=[ground_truth]
                    )
    trial = env.trial_sequence[0]
    num_actions = len(trial_actions)
    num_nodes = trial.num_nodes
    action_feature_values = np.zeros((num_actions, num_nodes, num_features))
    for i, action in enumerate(trial_actions):
        node_map = trial.node_map
        for node_num in range(num_nodes):
            node = trial.node_map[node_num]
            action_feature_values[i][node_num] = node.compute_termination_feature_values(
                                                                                features_list)
            if normalized_features:
                action_feature_values[i][node_num] = get_normalized_feature_values(
                action_feature_values[i][node_num], features_list, normalized_features)
        node_map[action].observe()
    return action_feature_values

def compute_error_gradient(w, trial_features, feature_indices, trial_actions, fit_inverse_temperature = False, compute_grad = False):
    grad = np.zeros_like(w)
    error = 0
    W = w
    beta = 1
    for i, action in enumerate(trial_actions):
        num_actions = trial_features[i].shape[0]
        available_actions = [a for a in range(num_actions) if a not in trial_actions[:i]]
        action_index = available_actions.index(action)
        required_features = trial_features[i][available_actions][:, feature_indices]
        dot_product = beta*np.dot(W, required_features.T)
        error += -dot_product[action_index] + logsumexp(dot_product)
        softmax_dot = softmax(dot_product)
        grad += np.dot(required_features.T, softmax_dot)
        grad += -1*required_features[action_index]
    return error, grad

def compute_total_error_gradient(w, simulated_features, feature_indices, simulated_actions,
                                fit_inverse_temperature = False, compute_grad = False):
    total_grad = np.zeros_like(w)
    total_error = 0
    num_simulations = len(simulated_actions)
    for sim_num in range(num_simulations):
        trial_actions = simulated_actions[sim_num]
        trial_features = simulated_features[sim_num]
        error, grad = compute_error_gradient(w, trial_features, feature_indices, trial_actions, fit_inverse_temperature, compute_grad)
        total_grad += grad
        total_error += error
    return total_error, total_grad

def get_termination_mers(envs, trial_actions, pipeline):
    mers = []
    for env, actions in zip(envs, trial_actions):
        trial = TrialSequence(1, pipeline, ground_truth = [env]).trial_sequence[0]
        for action in actions[:-1]:
            trial.node_map[action].observe()
        mers.append(trial.node_map[1].calculate_max_expected_return())
    return mers

class ClickSequence():
    def __init__(self, click_sequence, env, pipeline, features, normalized_features, feature_indices = None):
        self._click_sequence = click_sequence
        self._env = env
        self._pipeline = pipeline
        self._features = features
        self._num_features = len(features)
        self._normalized_features = normalized_features
        if not feature_indices:
            self._feature_indices = list(range(self._num_features))
        else:
            self._feature_indices = feature_indices
        self._feature_space = compute_trial_features(self._pipeline, self._env, self._click_sequence,
                                                     self._features, self._normalized_features)
    
    def compute_log_likelihoods(self, W, fit_temperatures = False):
        num_trials = 1
        trial = TrialSequence(1, pipeline = self._pipeline, ground_truth = [self._env]).trial_sequence[0]
        log_likelihoods = [compute_log_likelihood(trial, self._click_sequence, self._features,
                                                    w, fit_temperatures, self._normalized_features) for w in W]
        return log_likelihoods
    
    def compute_error_grad(self, w, compute_grad = True, fit_temperatures = False):
        res = compute_error_gradient(w, self._feature_space, self._feature_indices,
                                                 self._click_sequence, fit_temperatures, compute_grad)
        return res
