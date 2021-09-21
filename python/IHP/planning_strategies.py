import gym
from IHP.generic_mouselab import GenericMouselabEnv
from random import shuffle, choice
from queue import PriorityQueue
from IHP.modified_mouselab import TrialSequence, reward_val, approx_max, approx_min
from numpy import argsort

""" This file contains the 89 algorithmic strategies implemented for
    'Measuring how people learn how to plan'
"""

strategy_dict = {}


def strategy(name):
    def wrapper(func):
        strategy_dict[name] = func
    return wrapper


def get_second_max_dist_value(trial):
    values = []
    for d in range(1, trial.max_depth + 1):
        values.append(approx_max(trial.reward_function(d), position=1))
    return max(values)


def observe_till_root(trial, node):
    present_node = node
    while(present_node.parent.label != 0):
        if present_node.parent.observed:
            break
        present_node.parent.observe()
        present_node = present_node.parent


def observe_till_root_with_pruning(trial, node):
    present_node = node
    while(present_node.parent.label != 0):
        if present_node.parent.observed:
            return 0
        present_node.parent.observe()
        present_node = present_node.parent
        if present_node.value < 0 and not present_node.is_root():
            return 0
    return 1


def observe_randomly_till_root(trial, node):
    present_node = node
    nodes = []
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    shuffle(nodes)
    for node in nodes:
        if not node.observed:
            trial.node_map[node.label].observe()


def get_nodes_till_root(trial, node):
    present_node = node
    nodes = []
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    return nodes


def observe_path_from_root_to_node(trial, node):
    present_node = node
    nodes = [present_node]
    while(present_node.parent.label != 0):
        nodes.append(present_node.parent)
        present_node = present_node.parent
    nodes = nodes[::-1]
    for node in nodes:
        if not node.observed:
            trial.node_map[node.label].observe()


def observe_leaves_of_root(trial, root, satisficing_value=None):
    branch_nums = trial.reverse_branch_map[root.label]
    leaf_labels = []
    for branch_num in branch_nums:
        branch = trial.branch_map[branch_num]
        leaf_labels.append(branch[-1])
    shuffle(leaf_labels)
    for leaf in leaf_labels:
        node = trial.node_map[leaf]
        node.observe()
        if satisficing_value:
            if node.value >= satisficing_value:
                return 1
    return 0


def get_max_leaves(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    leaf_values = []
    for node in leaf_nodes:
        node.observe()
        leaf_values.append(node.value)
    max_leaf_value = max(leaf_values)
    best_nodes = [node for node in leaf_nodes if node.value == max_leaf_value]
    return best_nodes

def observe_nodes_satisficing(trial, nodes, satisficing_value):
    shuffle(nodes)
    satisficing_value_observed = False
    for node in nodes:
        node.observe()
        if node.value >= satisficing_value:
            satisficing_value_observed = True
            break
    if satisficing_value_observed:
        return 1
    else:
        return 0

def get_max_nodes(trial, nodes):
    node_values = [node.value for node in nodes]
    max_value = max(node_values)
    max_nodes = [node for node in nodes if node.value == max_value]
    shuffle(max_nodes)
    return max_nodes

def observe_node_path_subtree(trial, node, random=True):
    observe_till_root(trial, node)
    present_node = node
    while(present_node.parent.label != 0):
        present_node = present_node.parent
    path_root = present_node
    if random:
        successors = path_root.get_successor_nodes()
        unobserved_successors = [
            node for node in successors if not node.observed]
        shuffle(unobserved_successors)
        for node in unobserved_successors:
            node.observe()
    else:
        def observe_successors(node):
            if not node.children:
                return
            successors = node.get_successor_nodes()
            unobserved_successors = [
                node for node in successors if not node.observed]
            for child in unobserved_successors:
                if not child.observed:
                    child.observe()
                observe_successors(child)
        observe_successors(path_root)


def compare_paths_satisficing(trial, best_nodes):
    # Currently looks at all observed values and stops clicking when
    # a definite best is found
    temp_pointers = [node for node in best_nodes]
    best_node_values = [node.value for node in best_nodes]
    max_node_value = max(best_node_values)
    max_nodes = [node for node in best_nodes if node.value == max_node_value]
    if len(max_nodes) == 1:
        return
    while(1):
        parent_pointers = []
        parent_values = []
        for i, p in enumerate(temp_pointers):
            if p.parent.label != 0:
                if not temp_pointers[i].parent.observed:
                    temp_pointers[i].parent.observe()
                    parent_value = temp_pointers[i].parent.value
                    parent_pointers.append(temp_pointers[i].parent)
                    parent_values.append(parent_value)
        if parent_values:
            max_parent_value = max(parent_values)
            max_nodes = [
                node for node in parent_pointers if node.value == max_parent_value]
            if len(max_nodes) == 1:
                break
            temp_pointers = parent_pointers
        break


def get_top_root(node):
    if node.parent == node.root:
        return node
    else:
        temp = node
        while(temp.parent != node.root):
            temp = temp.parent
        return temp


def explore_subtree_random(subtree_root, threshold_value=None):
    successor_nodes = subtree_root.get_successor_nodes()
    if not subtree_root.observed:
        total_nodes = successor_nodes + [subtree_root]
    else:
        total_nodes = successor_nodes
    shuffle(total_nodes)
    if not threshold_value:
        for node in total_nodes:
            node.observe()
        return 0
    else:
        for node in total_nodes:
            node.observe()
            if node.value >= threshold_value:
                return 1


def get_subtree_leaves(subtree_root):
    successors = subtree_root.get_successor_nodes()
    successor_leaves = []
    for s in successors:
        if s.is_leaf():
            successor_leaves.append(s)
    return successor_leaves

def get_mid_nodes(trial):
    max_depth = trial.max_depth
    d = max_depth // 2
    nodes = trial.level_map[d + 1].copy()
    return nodes

def observe_nodes(nodes):
    for node in nodes:
        node.observe()

@strategy(1)
def goal_setting_random_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        node.observe()
        if node.value > 0:
            observe_randomly_till_root(trial, node)
    for node in leaf_nodes:
        observe_randomly_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(2)
def inverse_randomized_breadth_first_search(trial):
    max_depth = trial.max_depth
    for d in list(range(1, max_depth+1))[::-1]:
        nodes = trial.level_map[d]
        shuffle(nodes)
        observe_nodes(nodes)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(3)
def randomized_breadth_first_search(trial):
    max_depth = trial.max_depth
    for d in list(range(1, max_depth+1)):
        nodes = trial.level_map[d]
        shuffle(nodes)
        observe_nodes(nodes)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(4)
def progressive_deepening(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        observe_path_from_root_to_node(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(5)
# Pick according to best value starting from roots
def best_first_search_expected_value(trial):
    pq = PriorityQueue()
    pq.put((-0, 0))
    rf = trial.reward_function
    while not pq.empty():
        # pq.queue is not ordered according to priority. Only index 0 is right.
        top = trial.node_map[pq.queue[0][1]]
        best_child, best_child_value = None, -9999
        children = top.children.copy()
        shuffle(children)
        for child in children:
            if not child.observed:
                ev = rf(child.depth).expectation()
                if ev > best_child_value:
                    best_child = child
                    best_child_value = ev
        if best_child is None:
            pq.get()
            continue
        best_child.observe()
        pq.put((-best_child.value, best_child.label))
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(6)
def ancestor_priority(trial):
    """ Choose nodes according to the priority 0.8*number of
    ancestor nodes + 0.2*number of successor nodes
    """
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    while(len(unobserved_nodes) != 0):
        scores = []
        ancestor_scores = []
        for node in unobserved_nodes:
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.8*ancestor_count + 0.2*successor_count
            scores.append(score)
            ancestor_scores.append(ancestor_count)
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        max_ancestor_scores = [ancestor_scores[i] for i in max_indices]
        max_max_ancestor_scores = max(max_ancestor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i, s in enumerate(
            max_ancestor_scores) if s == max_max_ancestor_scores]
        node = choice(max_total_nodes)
        node.observe()
        unobserved_nodes.remove(node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(7)
def successor_priority(trial):
    """ Choose nodes according to the priority 0.8*number of
    successor nodes + 0.2*number of ancestor nodes
    """
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    while(len(unobserved_nodes) != 0):
        scores = []
        successor_scores = []
        for node in unobserved_nodes:
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.8*successor_count + 0.2*ancestor_count
            scores.append(score)
            successor_scores.append(successor_count)
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        max_successor_scores = [successor_scores[i] for i in max_indices]
        max_max_successor_scores = max(max_successor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i, s in enumerate(
            max_successor_scores) if s == max_max_successor_scores]
        node = choice(max_total_nodes)
        node.observe()
        unobserved_nodes.remove(node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(8)
def backward_path_goal_setting(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        node.observe()
        if node.value > 0:
            observe_till_root(trial, node)
    for node in leaf_nodes:
        observe_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(9)
def backward_path_subtree_goal_setting(trial):
    # Explores randomly in same subtree
    # We consider the nearest subtree
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        if not node.observed:
            node.observe()
            if node.value > 0:
                observe_node_path_subtree(trial, node)
    for node in leaf_nodes:
        observe_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(10)
def randomized_sibling_breadth_first_search(trial):
    max_depth = trial.max_depth
    for d in list(range(1, max_depth+1)):
        nodes = trial.level_map[d]
        shuffle(nodes)
        for node in nodes:
            if not node.observed:
                node.observe()
                siblings = node.get_sibling_nodes()
                shuffle(siblings)
                for sibling in siblings:
                    sibling.observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(11)
def immediate_ancestor_priority(trial):
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    while(len(unobserved_nodes) != 0):
        scores = []
        ancestor_scores = []
        for node in unobserved_nodes:
            immediate_ancestor_count = 1 if node.is_parent_observed() else 0
            immediate_successor_count = node.get_immediate_successor_count()
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.6*immediate_ancestor_count + 0.2*ancestor_count + (
                0.15*immediate_successor_count + 0.05*successor_count)
            scores.append(score)
            ancestor_scores.append(immediate_ancestor_count)
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        max_ancestor_scores = [ancestor_scores[i] for i in max_indices]
        max_max_ancestor_scores = max(max_ancestor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i, s in enumerate(
            max_ancestor_scores) if s == max_max_ancestor_scores]
        node = choice(max_total_nodes)
        node.observe()
        unobserved_nodes.remove(node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(12)
def immediate_successor_priority(trial):
    unobserved_nodes = trial.unobserved_nodes.copy()
    unobserved_nodes.remove(trial.node_map[0])
    shuffle(unobserved_nodes)
    while(len(unobserved_nodes) != 0):
        scores = []
        successor_scores = []
        for node in unobserved_nodes:
            immediate_ancestor_count = 1 if node.is_parent_observed() else 0
            immediate_successor_count = node.get_immediate_successor_count()
            ancestor_count = node.get_observed_ancestor_count()
            successor_count = node.get_observed_successor_count()
            score = 0.6*immediate_successor_count + 0.2*successor_count + (
                0.15*immediate_ancestor_count + 0.05*ancestor_count)
            scores.append(score)
            successor_scores.append(immediate_successor_count)
        max_score = max(scores)
        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        max_successor_scores = [successor_scores[i] for i in max_indices]
        max_max_successor_scores = max(max_successor_scores)
        max_total_nodes = [unobserved_nodes[max_indices[i]] for i, s in enumerate(
            max_successor_scores) if s == max_max_successor_scores]
        node = choice(max_total_nodes)
        node.observe()
        unobserved_nodes.remove(node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(13)
def backward_path_immediate_goal_setting(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        if not node.observed:
            node.observe()
            if node.value > 0:
                observe_node_path_subtree(trial, node, random=False)
    for node in leaf_nodes:
        observe_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(14)
def best_leaf_node_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    pq = PriorityQueue()
    for node in leaf_nodes:
        node.observe()
        pq.put((-node.value, node.label))
    while not pq.empty():
        _, node_num = pq.get()
        node = trial.node_map[node_num]
        observe_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(15)
def non_terminating_approximate_optimal(trial):
    # Similar to 16 but doesn't terminate
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    leaf_values = []
    for node in leaf_nodes:
        node.observe()
        leaf_values.append(node.value)
        if node.value >= max_value:
            observe_till_root(trial, node)
    indices = argsort(leaf_values)[::-1]
    for i in indices:
        observe_till_root(trial, leaf_nodes[i])
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(16)
def goal_setting_backward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            observe_till_root(trial, node)
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(17)
def goal_setting_equivalent_goals_level_by_level(trial):
    best_nodes = get_max_leaves(trial)
    # Check if this still holds in case of unequal path lengths
    temp_pointers = best_nodes
    while(1):
        parent_pointers = []
        for i, p in enumerate(temp_pointers):
            if p.parent.label != 0:
                if not temp_pointers[i].parent.observed:
                    temp_pointers[i].parent.observe()
                    parent_pointers.append(temp_pointers[i].parent)
        if len(parent_pointers) == 0:
            return [node.label for node in trial.observed_nodes] + [0]
        temp_pointers = parent_pointers
        shuffle(temp_pointers)

@strategy(18)
def goal_setting_equivalent_goals_random(trial):
    best_nodes = get_max_leaves(trial)
    shuffle(best_nodes)
    nodes_list = []
    for node in best_nodes:
        nodes_list += get_nodes_till_root(trial, node)
    nodes_list = list(set(nodes_list))
    shuffle(nodes_list)
    for node in nodes_list:
        node.observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(19)
def goal_setting_equivalent_goals_leaf_to_root(trial):
    best_nodes = get_max_leaves(trial)
    shuffle(best_nodes)
    for node in best_nodes:
        observe_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(20)
def goal_setting_equivalent_goals_root_to_leaf(trial):
    best_nodes = get_max_leaves(trial)
    shuffle(best_nodes)
    for node in best_nodes:
        observe_path_from_root_to_node(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(21)
def approximate_optimal(trial):
    max_value = trial.get_max_dist_value()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    required_nodes = leaf_nodes
    satisficing_observed = observe_nodes_satisficing(trial, required_nodes, max_value)
    max_depth = trial.max_depth
    while not satisficing_observed and max_depth != 0:
        max_depth -= 1
        max_nodes = get_max_nodes(trial, required_nodes)
        required_nodes = list(set([node.parent for node in max_nodes]))
        if len(required_nodes) == 1:
            break
        shuffle(required_nodes)
        max_value = trial.max_values_by_depth[max_depth]
        satisficing_observed = observe_nodes_satisficing(trial, required_nodes, max_value)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(22)
def positive_forward_satisficing(trial):
    """ Terminate on finding positive root.
        If no positive root node is found,
        terminates after exploring all root nodes
    """
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        node.observe()
        if node.value > 0:
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(23)
def check_all_roots(trial):
    """ Explores all root nodes and terminates """
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        node.observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(24)
def check_all_leaves(trial):
    """ Explores all leaf nodes and terminates """
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        node.observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(25)
def satisficing_best_first_search_expected_value(trial):
    pq = PriorityQueue()
    pq.put((-0, 0))
    rf = trial.reward_function
    max_value = trial.get_max_dist_value()
    while not pq.empty():
        # pq.queue is not ordered according to priority. Only index 0 is right.
        top = trial.node_map[pq.queue[0][1]]
        best_child, best_child_value = None, -9999
        children = top.children.copy()
        shuffle(children)
        for child in children:
            if not child.observed:
                ev = rf(child.depth).expectation()
                if ev > best_child_value:
                    best_child = child
                    best_child_value = ev
        if best_child is None:
            pq.get()
            continue
        best_child.observe()
        if best_child.value >= max_value:
            break
        pq.put((-best_child.value, best_child.label))
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(26)
def backward_positive_satisficing(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        node.observe()
        if node.value > 0:
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(27)
def one_final_outcome(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    leaf_nodes[0].observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(28)
def one_immediate_outcome(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    root_nodes[0].observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(29)
def goal_setting_forward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            observe_path_from_root_to_node(trial, node)
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(30)
def no_planning(trial):
    return [0]


@strategy(31)
def satisficing_dfs(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()

    def dfs(node):
        node.observe()
        if node.value >= max_value:
            return 1
        if not node.children:
            return 0
        else:
            for child in node.children:
                res = dfs(child)
                if res == 1:
                    return 1
    shuffle(root_nodes)
    for root_node in root_nodes:
        res = dfs(root_node)
        if res == 1:
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(32)
def positive_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    observe_nodes(root_nodes)
    for root in root_nodes:
        if root.value > 0:
            observe_leaves_of_root(trial, root)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(33)
def satisficing_positive_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    max_value = trial.get_max_dist_value()
    observe_nodes(root_nodes)
    for root in root_nodes:
        if root.value > 0:
            res = observe_leaves_of_root(
                trial, root, satisficing_value=max_value)
            if res == 1:
                break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(34)
def positive_satisficing_positive_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for root in root_nodes:
        root.observe()
        if root.value >= 0:
            observe_leaves_of_root(trial, root, satisficing_value=0)
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(35)
def backward_planning_positive_outcomes(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for leaf in leaf_nodes:
        leaf.observe()
        if leaf.value > 0:
            observe_till_root(trial, leaf)
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(36)
def best_first_search_observed_roots(trial):
    # Currently implemented as starting after looking at all roots
    pq = PriorityQueue()
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        node.observe()
        pq.put((-node.value, node.label))
    while(not pq.empty()):
        _, node_num = pq.get()
        node = trial.node_map[node_num]
        children = node.children.copy()
        if children:
            shuffle(children)
            for child in children:
                child.observe()
                pq.put((-child.value, child.label))
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(37)
def satisficing_best_first_search_observed_roots(trial):
    # Currently implemented as starting after looking at all roots
    pq = PriorityQueue()
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    max_value = trial.get_max_dist_value()
    for node in root_nodes:
        node.observe()
        if node.value >= max_value:
            return [node.label for node in trial.observed_nodes] + [0]
        pq.put((-node.value, node.label))
    while(not pq.empty()):
        _, node_num = pq.get()
        node = trial.node_map[node_num]
        children = node.children.copy()
        if children:
            shuffle(children)
            for child in children:
                child.observe()
                if child.value >= max_value:
                    return [node.label for node in trial.observed_nodes] + [0]
                pq.put((-child.value, child.label))
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(38)
def loss_averse_backward_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    max_value = trial.get_max_dist_value()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            status = observe_till_root_with_pruning(trial, node)
            if status == 1:
                break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(39)
def random_planning(trial):
    nodes = list(trial.node_map.values())
    shuffle(nodes)
    node_labels = [node.label for node in nodes]
    termination_index = node_labels.index(0)
    return node_labels[:termination_index+1]


@strategy(40)
def extra_optimal_planning(trial):
    max_value = trial.get_max_dist_value()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    count = 0
    for node in leaf_nodes:
        node.observe()
        if count == 1:
            break
        if node.value >= max_value:
            count += 1
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(41)
def leave_out_one_leaf(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes[:-1])
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(42)
def explore_roots_non_optimal(trial):
    max_value = trial.get_max_dist_value()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes)
    max_found = False
    for node in leaf_nodes:
        if node.value >= max_value:
            max_found = True
            break
    shuffle(leaf_nodes)
    if not max_found:
        max_value = get_second_max_dist_value(trial)
        for node in leaf_nodes:
            if node.value >= max_value:
                root = get_top_root(node)
                if not root.observed:
                    root.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(43)
def consecutive_second_extra(trial):
    leaf_nodes = trial.get_leaf_nodes()
    second_best = get_second_max_dist_value(trial)
    previous = False
    count = 0
    for node in leaf_nodes:
        node.observe()
        if count == 1:
            break
        if node.value >= second_best:
            if previous:
                count += 1
            else:
                previous = True
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(44)
def explore_one_subtree_random(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    subtree_root = root_nodes[0]
    explore_subtree_random(subtree_root)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(45)
def leave_one_subtree_random(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for subtree_root in root_nodes[:-1]:
        explore_subtree_random(subtree_root)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(46)
def explore_all_subtrees_random(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for subtree_root in root_nodes:
        explore_subtree_random(subtree_root)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(47)
def explore_all_subtrees_termination(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    max_value = trial.get_max_dist_value()
    for subtree_root in root_nodes:
        res = explore_subtree_random(subtree_root, max_value)
        if res == 1:
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(48)
def explore_roots_and_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(root_nodes)
    shuffle(leaf_nodes)
    observe_nodes(root_nodes)
    observe_nodes(leaf_nodes)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(49)
def explore_roots_leaves_termination(trial):
    root_nodes = trial.node_map[0].children.copy()
    leaf_nodes = trial.get_leaf_nodes()
    max_value = trial.get_max_dist_value()
    shuffle(root_nodes)
    shuffle(leaf_nodes)
    observe_nodes(root_nodes)
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(50)
def one_above_max(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            node.parent.observe()
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(51)
def root_of_max(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            max_root = get_top_root(node)
            max_root.observe()
            break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(52)
def extra_positive_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    count = 0
    for node in leaf_nodes:
        node.observe()
        if count == 1:
            break
        if node.value > 0:
            count += 1
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(53)
def leave_one_root(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    observe_nodes(root_nodes[:-1])
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(54)
def explore_roots_subtrees(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    observe_nodes(root_nodes)
    shuffle(root_nodes)
    for node in root_nodes:
        explore_subtree_random(node)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(55)
def explore_num_roots_leaves(trial):
    num_roots = len(trial.node_map[0].children)
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes[:num_roots])
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(56)
def explore_different_subtree_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        successor_leaves = get_subtree_leaves(node)
        shuffle(successor_leaves)
        successor_leaves[0].observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(57)
def consecutive_second(trial):
    leaf_nodes = trial.get_leaf_nodes()
    second_best = get_second_max_dist_value(trial)
    previous = False
    for node in leaf_nodes:
        node.observe()
        if node.value >= second_best:
            if previous:
                break
            previous = True
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(58)
def double_extra_positive_planning(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    count = 0
    for node in leaf_nodes:
        node.observe()
        if count == 2:
            break
        if node.value > 0:
            count += 1
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(59)
def one_subtree_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    explore_subtree_random(root_nodes[0])
    leaves = get_subtree_leaves(root_nodes[1])
    shuffle(leaves)
    observe_nodes(leaves)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(60)
def double_subtree_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    explore_subtree_random(root_nodes[0])
    explore_subtree_random(root_nodes[1])
    leaves = get_subtree_leaves(root_nodes[2])
    shuffle(leaves)
    observe_nodes(leaves)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(61)
def explore_one_path(trial):
    branches = list(trial.branch_map.values())
    selected_branch = choice(branches)
    for node_num in selected_branch[1:]:
        node = trial.node_map[node_num]
        node.observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(62)
def explore_leaves_and_roots(trial):
    root_nodes = trial.node_map[0].children.copy()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(root_nodes)
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes)
    observe_nodes(root_nodes)
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(63)
def explore_leaves_and_roots_termination(trial):
    root_nodes = trial.node_map[0].children.copy()
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(root_nodes)
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    max_found = False
    for node in leaf_nodes:
        node.observe()
        if node.value >= max_value:
            max_found = True
            break
    if not max_found:
        for node in root_nodes:
            node.observe()
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(64)
def root_leaves_termination(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()
    shuffle(root_nodes)
    max_found = False
    for node in root_nodes:
        node.observe()
        if node.value >= 0:
            successor_leaves = get_subtree_leaves(node)
            shuffle(successor_leaves)
            for leaf in successor_leaves:
                leaf.observe()
                if leaf.value >= max_value:
                    max_found = True
                    break
    if not max_found:
        leaf_nodes = trial.get_leaf_nodes()
        shuffle(leaf_nodes)
        for leaf in leaf_nodes:
            if not leaf.observed:
                leaf.observe()
                if leaf.value >= max_value:
                    break
    return [node.label for node in trial.observed_nodes] + [0]


@strategy(65)
def root_leaves_termination_positive(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = 0
    shuffle(root_nodes)
    max_found = False
    for node in root_nodes:
        node.observe()
        if node.value >= 0:
            successor_leaves = get_subtree_leaves(node)
            shuffle(successor_leaves)
            for leaf in successor_leaves:
                leaf.observe()
                if leaf.value >= max_value:
                    max_found = True
                    break
    if not max_found:
        leaf_nodes = trial.get_leaf_nodes()
        shuffle(leaf_nodes)
        for leaf in leaf_nodes:
            if not leaf.observed:
                leaf.observe()
                if leaf.value >= max_value:
                    break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(66)
def explore_single_mid_node(trial):
    mid_nodes = get_mid_nodes(trial)
    shuffle(mid_nodes)
    mid_nodes[0].observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(67)
def best_leaf_single_node_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes)
    max_value = max([node.value for node in leaf_nodes])
    max_nodes = [node for node in leaf_nodes if node.value == max_value]
    shuffle(max_nodes)
    for node in max_nodes:
        if not node.parent.observed:
            node.parent.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(68)
def best_leaf_complete_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes)
    max_value = max([node.value for node in leaf_nodes])
    max_nodes = [node for node in leaf_nodes if node.value == max_value]
    shuffle(max_nodes)
    for node in max_nodes:
        observe_till_root(trial, node)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(69)
def best_subtree(trial):
    root_nodes = trial.node_map[0].children.copy()
    observe_nodes(root_nodes)
    max_value = max([node.value for node in root_nodes])
    max_nodes = [node for node in root_nodes if node.value == max_value]
    selected_node = choice(max_nodes)
    explore_subtree_random(selected_node)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(70)
def best_root_child(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    observe_nodes(root_nodes)
    max_value = max([node.value for node in root_nodes])
    max_nodes = [node for node in root_nodes if node.value == max_value]
    selected_node = choice(max_nodes)
    children = selected_node.children
    selected_child = choice(children)
    selected_child.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(71)
def root_leaf_mid(trial):
    root_nodes = trial.node_map[0].children.copy()
    leaf_nodes = trial.get_leaf_nodes()
    mid_nodes = get_mid_nodes(trial)
    shuffle(root_nodes)
    shuffle(leaf_nodes)
    shuffle(mid_nodes)
    observe_nodes(root_nodes)
    observe_nodes(leaf_nodes)
    observe_nodes(mid_nodes)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(72)
def mid_root_leaf(trial):
    root_nodes = trial.node_map[0].children.copy()
    leaf_nodes = trial.get_leaf_nodes()
    mid_nodes = get_mid_nodes(trial)
    shuffle(root_nodes)
    shuffle(leaf_nodes)
    shuffle(mid_nodes)
    observe_nodes(mid_nodes)
    observe_nodes(root_nodes)
    observe_nodes(leaf_nodes)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(73)
def explore_two_leaves(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes[:2])
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(74)
def explore_all_mid_nodes(trial):
    mid_nodes = get_mid_nodes(trial)
    shuffle(mid_nodes)
    observe_nodes(mid_nodes)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(75)
def explore_best_leaf_roots(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    observe_nodes(leaf_nodes)
    max_value = max([node.value for node in leaf_nodes])
    max_nodes = [node for node in leaf_nodes if node.value == max_value]
    shuffle(max_nodes)
    for node in max_nodes:
        root = get_top_root(node)
        if not root.observed:
            root.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(76)
def explore_random_max_path(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    max_value = trial.get_max_dist_value()
    for leaf in leaf_nodes:
        leaf.observe()
        if leaf.value >= max_value:
            observe_till_root(trial, leaf)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(77)
def positive_extra_leaf(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    count = 0
    for node in leaf_nodes:
        node.observe()
        if count == 1:
            break
        if node.value > 0:
            count += 1
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(78)
def positive_parent_leaf(trial):
    leaf_nodes = trial.get_leaf_nodes()
    shuffle(leaf_nodes)
    for node in leaf_nodes:
        node.observe()
        if node.value > 0:
            node.parent.observe()
            break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(79)
def explore_positive_subtrees(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    observe_nodes(root_nodes)
    positive_nodes = [node for node in root_nodes if node.value > 0]
    shuffle(positive_nodes)
    for node in positive_nodes:
        explore_subtree_random(node)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(80)
def general_root_leaves(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()
    shuffle(root_nodes)
    max_found = False
    for node in root_nodes:
        if max_found:
            break
        node.observe()
        successor_leaves = get_subtree_leaves(node)
        shuffle(successor_leaves)
        for leaf in successor_leaves:
            leaf.observe()
            if leaf.value >= max_value:
                max_found = True
                break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(81)
def positive_root_path(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        node.observe()
        if node.value > 0:
            successor_leaves = get_subtree_leaves(node)
            chosen_leaf = choice(successor_leaves)
            observe_path_from_root_to_node(trial, chosen_leaf)
            if chosen_leaf.value > 0:
                break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(82)
def breadth_first_search_termination(trial):
    max_depth = trial.max_depth
    max_value = trial.get_max_dist_value()
    for d in range(1, max_depth + 1):
        nodes = trial.level_map[d].copy()
        shuffle(nodes)
        if d != max_depth:
            observe_nodes(nodes)
        else:
            for node in nodes:
                node.observe()
                if node.value >= max_value:
                    break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(83)
def explore_siblings(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()
    shuffle(root_nodes)
    max_found = False
    for node in root_nodes:
        if max_found:
            break
        successor_leaves = get_subtree_leaves(node)
        shuffle(successor_leaves)
        for leaf in successor_leaves:
            leaf.observe()
            if leaf.value >= max_value:
                observe_path_from_root_to_node(trial, leaf)
                max_found = True
                break
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(84)
def explore_subtree_mid(trial):
    root_nodes = trial.node_map[0].children.copy()
    max_value = trial.get_max_dist_value()
    shuffle(root_nodes)
    max_found = False
    for root in root_nodes:
        if max_found:
            selected_child = choice(root.children)
            if not selected_child.observed:
                selected_child.observe()
            break
        total_nodes = [root] + root.get_successor_nodes()
        shuffle(total_nodes)
        for node in total_nodes:
            node.observe()
            if node.value >= max_value:
                max_found = True
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(85)
def explore_center_leaves(trial):
    leaf_nodes = trial.get_leaf_nodes()
    mid_nodes = get_mid_nodes(trial)
    shuffle(leaf_nodes)
    shuffle(mid_nodes)
    observe_nodes(mid_nodes)
    observe_nodes(leaf_nodes)
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(86)
def mid_subtree_leaf(trial):
    mid_nodes = get_mid_nodes(trial)
    shuffle(mid_nodes)
    for node in mid_nodes:
        node.observe()
        selected_child = choice(node.children)
        selected_child.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(87)
def leaves_middle_subtrees(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    for node in root_nodes:
        leaves = get_subtree_leaves(node)
        shuffle(leaves)
        observe_nodes(leaves)
        selected_leaf = choice(leaves)
        selected_leaf.parent.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(88)
def leaves_middle_subtrees_satisficing(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    max_value = trial.get_max_dist_value()
    max_found = False
    for node in root_nodes:
        if max_found:
            break
        leaves = get_subtree_leaves(node)
        shuffle(leaves)
        for leaf in leaves:
            leaf.observe()
            if leaf.value >= max_value:
                max_found = True
                break
        if not max_found:
            selected_leaf = choice(leaves)
            selected_leaf.parent.observe()
    return [node.label for node in trial.observed_nodes] + [0]

@strategy(89)
def leaves_middle_subtree(trial):
    root_nodes = trial.node_map[0].children.copy()
    shuffle(root_nodes)
    node = root_nodes[0]
    leaves = get_subtree_leaves(node)
    shuffle(leaves)
    observe_nodes(leaves)
    selected_leaf = choice(leaves)
    selected_leaf.parent.observe()
    return [node.label for node in trial.observed_nodes] + [0]

if __name__ == "__main__":
    from learning_utils import Participant
    p = Participant(exp_num='T1.1', pid=1,
                    feature_criterion='normalize', excluded_trials=list(range(11)))
    trials = p.envs
    num_trials = len(trials)
    env = TrialSequence(
        num_trials,
        pipeline,
        ground_truth=trials
    )
    for strategy in range(1, 40):
        trial = env.trial_sequence[0]
        trial.reset_observations()
        print(strategy)
        print(strategy_dict[strategy](trial))
