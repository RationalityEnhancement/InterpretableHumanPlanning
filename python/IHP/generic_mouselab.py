import gym
import numpy as np
from gym import spaces
from random import choice
from IHP.modified_mouselab import TrialSequence, reward_val


class GenericMouselabEnv(gym.Env):
    """
        This class is the gym environment for the feature based version
        of the Mouselab-MDP. The environment structure is assumed to be a
        symmetric tree
    """
    def __init__(self, num_trials=1, pipeline = {'0': ([3,1,2], reward_val)},
				ground_truth=None, cost=1, render_path="mouselab_renders"):
        super(GenericMouselabEnv, self).__init__()
        self.pipeline = pipeline
        self.ground_truth = ground_truth
        self.cost = cost
        self.num_trials = num_trials
        self.render_path = render_path
        self.repeat_cost = -self.cost*10
        self.construct_env()

    def custom_same_env_init(self, env, num_trials):
        self.num_trials = num_trials
        ground_truths = [env]*self.num_trials
        self.ground_truth = ground_truths
        self.construct_env()

    def participant_init(self, ground_truth):
        self.num_trials = len(ground_truth)
        self.ground_truth = ground_truth
        self.construct_env()

    def construct_env(self):
        self.trial_sequence = TrialSequence(self.num_trials, self.pipeline,
                                            self.ground_truth)
        self.present_trial_num = 0
        self.trial_init()
        if not self.ground_truth:
            self.ground_truth = self.trial_sequence.ground_truth
    
    def trial_init(self):
        trial_num = self.present_trial_num
        self.num_nodes = len(self.trial_sequence.trial_sequence[trial_num].node_map)
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(self.num_nodes,), dtype=np.float32)
        self.present_trial = self.trial_sequence.trial_sequence[trial_num]
        reward_function = self.pipeline[self.present_trial_num][1]
        self.node_distribution = [
            [0]] + [reward_function(d) for d in range(1, self.present_trial.max_depth + 1)]
        self._compute_expected_values()
        self._construct_state()
        self.observed_action_list = []

    def _construct_state(self):
        self._state = [0] + [self.node_distribution[self.present_trial.node_map[node_num].depth]
                             for node_num in range(1, self.num_nodes)]

    def _compute_expected_values(self):
        self.expected_values = [0] + [self.node_distribution[self.present_trial.node_map[node_num].depth].expectation()
                                      for node_num in range(1, self.num_nodes)]

    def get_next_trial(self):
        if self.present_trial_num == self.num_trials - 1:
            return -1
        self.present_trial_num += 1
        self.trial_init()
        self.observed_action_list = []
        return None

    def reset_trial(self):
        self._compute_expected_values()
        self._construct_state()
        self.present_trial.reset_observations()
        self.observed_action_list = []

    def reset(self):
        self.construct_env()
        return self._state

    def step(self, action):
        info = {}
        reward = -self.cost
        done = False
        if action in self.observed_action_list:
            return self._state, self.repeat_cost, False, {}
        self.observed_action_list.append(action)
        node_map = self.present_trial.node_map
        if not action == 0:
            self.present_trial.node_map[action].observe()
        else:
            done = True
            best_expected_path = self.present_trial.get_best_expected_path()
            info = best_expected_path[1:]
            reward = 0
            for node in best_expected_path:
                reward += self.present_trial.node_map[node].value
                # self.present_trial.node_map[node].observe()
        self._state[action] = node_map[action].value
        return self._state, reward, done, info

    def render(self, dir_path=None):
        pass

    def get_random_env(self):
        trial_sequence = TrialSequence(num_trials=1, pipeline = self.pipeline)
        return trial_sequence.ground_truth[0]

    def get_ground_truth(self):
        return self.ground_truth

    def get_available_actions(self):
        num_list = [i for i in range(
            self.num_nodes) if i not in self.observed_action_list]
        return num_list

class ModStateGenericMouselabEnv(GenericMouselabEnv):
    def __init__(self, num_trials=1, pipeline = {'0': ([3,1,2], reward_val)},
				ground_truth=None, cost=1, render_path="mouselab_renders"):
        super().__init__(num_trials, pipeline, ground_truth, cost, render_path)

    def _construct_state(self):
        self._state = np.array([0] + [0 for node_num in range(1, self.num_nodes)])

    def step(self, action):
        S, reward, done, info = super().step(action)
        for node in self.present_trial.node_map.keys():
            if node not in self.observed_action_list:
                S[node] = 0
        S = np.array(S)
        return S, reward, done, info

class DummyParticipant():
    """ Creates a participant object which contains all details about the participant

    Returns:
        Participant -- Contains details such as envs, scores, clicks, taken paths,
                       strategies and weights at each trial.
    """

    def __init__(self, pipeline, num_trials):
        self.all_trials_data = self.get_all_trials_data()
        self.num_trials = num_trials
        self.pipeline = pipeline

    @property #This is done on purpose to induce stochasticity
    def envs(self):
        envs = GenericMouselabEnv(self.num_trials, self.pipeline).ground_truth
        self.trial_envs = envs
        return envs

    def get_envs(self):
        return self.trial_envs

    def get_all_trials_data(self):
        total_data = {'actions': {}, 'rewards': {},
                      'taken_paths': {}, 'strategies': {},
                      'temperature':{}}
        return total_data
