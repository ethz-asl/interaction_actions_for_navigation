from __future__ import print_function
import copy
from heapq import heappush, heappop
from CMap2D import gridshow
import matplotlib.pyplot as plt
import numpy as np
from pyniel.pyplot_tools.realtime import plot_closeall_button
import warnings

import pyIA.paths as paths
from cia import segment_endindex_at_pathpos_i
from cia import path_state_from_state_and_path
from cia import sultans_wife
from cia import bool1d_from_segments
from cia import CTaskState
from cia import kStateFeatures

# Planner
ACTION_CMAP = plt.cm.rainbow
ACTION_LENGTH_LIMIT_IN_TASK_TREE = 6.  # meters

# STATE_FEATURES = ... # list of state features has been moved to cia.pyx
DYNAMIC_STATE = False  # the whole state is dynamic (useful only if paths can change within MCTS)
DYNAMIC_PATH_STATE = True  # only the path state is dynamic
STORE_ESTIMATE_DISTRIBUTIONS = True
ALLOW_CONSECUTIVE_FAILURE = True



class Branch(object):
    def __init__(self, target, action, meta={
            "success": None,
            "cost": None,
            "prob": None,
    }):
        self.target = target  # id of node at the branch end
        self.action = action
        self.meta = meta


class BranchBundle(object):
    def __init__(self, meta={
            "tasktargetpathidx": None,
            "taskaction": None,
            "taskfullpath": None,
    }):
        self.branches = []  # [Branch(), ...]
        self.meta = meta


class TTNode(object):
    def __init__(self):
        self.childbranches = set()


class STNode(object):
    def __init__(self,
                 next_id,
                 depth,
                 startidx,
                 state,
                 path_state,
                 path_variant,
                 task_tree,
                 possible_actions,
                 cumulative_cost=None,
                 cumulative_prob=None,
                 ):
        # metadata
        self.id = next_id[0]  # "pass by reference" hack
        next_id[0] += 1
        self.depth = depth
        self.is_open = True
        # data
        self.startidx = startidx  # along path
        # edges
        self.childbranchbundles = []
        # indirect data
        self.cumulative_cost = cumulative_cost
        self.cumulative_prob = cumulative_prob
        # meta
        self.mce_info = None  # monte carlo estimate info can be stored here as NodeMCEstimateInfo()
        # necessary for lazy expansion (should only be set if node is open and yet to be updated)
        self.task_state_update = None
        # lazy expansion data (not know until the node is closed)
        self.is_terminal = False
        self.state = state
        self.path_state = path_state
        self.path_variant = path_variant
        self.task_tree = task_tree
        self.possible_actions = possible_actions

    def copy(self):
        # only the childbranchbundles are deepcopied! nodes are generally not copied.
        copy_ = STNode(
            [self.id],
            self.depth,
            self.startidx,
            self.state.copy(),
            self.path_state,  # TODO copy()
            self.path_variant,
            self.task_tree,
            self.possible_actions,
        )
        copy_.is_open = self.is_open
        copy_.childbranchbundles = copy.deepcopy(self.childbranchbundles)
        copy_.cumulative_cost = self.cumulative_cost
        copy_.cumulative_prob = self.cumulative_prob
        copy_.mce_info = copy.deepcopy(self.mce_info)
        copy_.task_state_update = self.task_state_update
        copy_.is_terminal = self.is_terminal
        return copy_


class MCBranchEstimate(object):
    def __init__(self):
        self.n_goal_reached = 0  # times goal was reached when this node/branch was selected
        self.n_total = 0  # times this node/branch was selected
        self.sum_cost_to_goal = 0  # total final cost of all trials where this node/branch was selected
        # optional, store distribution of outcome costs
        self.dist = []

    def expected_final_cost(self):
        if self.n_goal_reached == 0:
            return np.nan
        return 1. * self.sum_cost_to_goal / self.n_goal_reached

    def prob_reach_goal(self):
        return 1. * self.n_goal_reached / self.n_total


class NodeMCEstimateInfo(object):
    def __init__(self, option_mc_estimates):
        self.option_mc_estimates = option_mc_estimates
        self.endnode_estimate = MCBranchEstimate()  # estimate for when MC algo ends on this node

    def node_estimate(self):
        mc_estimate = MCBranchEstimate()
        # sum branch estimates
        for est in self.option_mc_estimates:
            mc_estimate.n_goal_reached += est.n_goal_reached
            mc_estimate.n_total += est.n_total
            mc_estimate.sum_cost_to_goal += est.sum_cost_to_goal
            mc_estimate.dist.extend(est.dist)
        # add terminal estimates for self
        mc_estimate.n_goal_reached += self.endnode_estimate.n_goal_reached
        mc_estimate.n_total += self.endnode_estimate.n_total
        mc_estimate.sum_cost_to_goal += self.endnode_estimate.sum_cost_to_goal
        mc_estimate.dist.extend(self.endnode_estimate.dist)
        return mc_estimate


class StochasticTree(object):
    def __init__(self):
        self.nodes = {}
        # meta
        self.root_nodes = []
        self.next_id = [0]
        self.fixed_state = None


# CORE interface -----------
def path_options(state, fixed_state, possible_path_variants=None, debug=False):
    if possible_path_variants is None:
        possible_path_variants = paths.PATH_VARIANTS
    paths_xy = []
    path_variants = []
    for variant in possible_path_variants:
        path_xy = fixed_state.derived_state.path_to_goal(state, fixed_state, variant)
        if path_xy is None:
            continue
        if paths.is_path_in_list(path_xy, paths_xy):
            continue
        paths_xy.append(path_xy)
        path_variants.append(variant)
    if debug:
        plt.show()
    return [np.array(path, dtype=np.float32) for path in paths_xy], path_variants

def task_tree_from_path_state(path_state, possible_actions,
                              task_length_limit=None, constrain_to_depth_1=False):
    path_len = path_state.get_path_len()
    action_segment_options = []
    # check action preconditions
    for action in possible_actions:
        proto_segments, segment_constraints = action.check_preconditions_on_path(path_state)
        proto_endindices = segment_endindex_at_pathpos_i(proto_segments, path_state)
        action_segment_options.append(
            (action, proto_segments, proto_endindices, segment_constraints)
        )
    # trigger points
    trigger_points = {-1}  # -1 is a convenience to point to the segment end
    for option in action_segment_options:
        _, proto_segments, _, _ = option
        for seg in proto_segments:
            if seg[0] != 0:
                trigger_points.add(seg[0])
    # build tree by combining valid action proto-segments
    firstidx = path_state.pos_along_path
    open_nodes = {firstidx}  # set {nodeindex}
    closed_nodes = {path_len - 1: TTNode()}  # dict {nodeindex: TTNode}
    while True:
        if not open_nodes:
            break
        actionstartidx = open_nodes.pop()
        node = TTNode()
        for option in action_segment_options:
            action, _, proto_endindices, segment_constraints = option
            is_length_limit_trigger_point_added = False  # avoid putting the same trigger point twice
            # check if action can be started at this point in path
            if proto_endindices[actionstartidx] == -1:
                continue
            # multiple possibilities for ending: end at segment end or trigger point
            for t in trigger_points:
                if t == -1:
                    actionendidx = proto_endindices[actionstartidx]
                # only allow triggerpoints in the future
                elif t <= actionstartidx:
                    continue
                # don't allow triggerpoints outside of action preconditions
                elif t - 1 > proto_endindices[actionstartidx]:
                    continue
                else:
                    # set action to end right before the trigger point
                    # this way the next action will start AT the trigger point
                    actionendidx = t - 1
                # add hacky length constraint
                if task_length_limit is not None:
                    if actionendidx > actionstartidx + task_length_limit:
                        if is_length_limit_trigger_point_added:
                            continue
                        is_length_limit_trigger_point_added = True
                        actionendidx = actionstartidx + task_length_limit
                # check if constraints are held by taking this action
                if not segment_constraints((actionstartidx, actionendidx), path_state):
                    continue
                # add action to the tree
                resultidx = actionendidx + 1
                if resultidx >= path_len:
                    resultidx = path_len - 1
                node.childbranches.add(Branch(resultidx, action))
                if resultidx not in closed_nodes:
                    open_nodes.add(resultidx)
        closed_nodes[actionstartidx] = node
        if constrain_to_depth_1:
            for node_startidx in open_nodes:
                closed_nodes[node_startidx] = TTNode()
            break
    task_tree = closed_nodes
    return task_tree

def plan(state, fixed_state, possible_actions,
         n_trials=1000, max_depth=10,
         BYPRODUCTS=None, DEBUG=False, INTERRUPT=None, DEBUG_IF_FAIL=True):
    byproducts = {}
    if BYPRODUCTS is not None:
        byproducts = BYPRODUCTS
    paths_xy, path_variants = path_options(state, fixed_state)
    byproducts["paths_xy"] = paths_xy
    byproducts["path_variants"] = path_variants
    path_states = [path_state_from_state_and_path(state, path_xy, fixed_state.map.xy_to_floatij(path_xy))
                   for path_xy in paths_xy]
    byproducts["path_states"] = path_states
    task_length_limit = None
    if ACTION_LENGTH_LIMIT_IN_TASK_TREE is not None:
        task_length_limit = int(ACTION_LENGTH_LIMIT_IN_TASK_TREE / fixed_state.map.resolution())
    task_trees = [task_tree_from_path_state(path_state, possible_actions, task_length_limit)
                  for path_state in path_states]
    byproducts["task_trees"] = task_trees
    stochastic_tree = initialize_stochastic_tree(
        state, fixed_state, path_states, path_variants, task_trees, possible_actions, DEBUG=DEBUG)
    byproducts["stochastic_tree"] = stochastic_tree
    # Detect Error
    if not stochastic_tree.root_nodes:
        if DEBUG_IF_FAIL:
            print("Error: no possibilities found.")
            visualize_state_features(state, fixed_state, hide_uncertain=False)
            visualize_traversabilities(state, fixed_state, paths.PATH_VARIANTS)
            visualize_dijkstra_fields(state, fixed_state, paths.PATH_VARIANTS)
            plt.show()
            return StochasticTree(), [], []
    monte_carlo_tree_search(
        stochastic_tree, n_trials=n_trials, max_depth=max_depth, DEBUG=DEBUG, INTERRUPT=INTERRUPT)
    # Planning outputs
    if INTERRUPT is not None:
        if INTERRUPT[0]:
            return None, None, None
    pruned_stochastic_tree = greedy_prune_tree(stochastic_tree)
    optimistic_sequence = optimistic_sequence_from_pruned_tree(pruned_stochastic_tree)
    firstfailure_sequence = firstfailure_then_optimistic_sequence_from_pruned_tree(pruned_stochastic_tree)
    return pruned_stochastic_tree, optimistic_sequence, firstfailure_sequence

def initialize_stochastic_tree(
        state, fixed_state, path_states, path_variants, task_trees, possible_actions, DEBUG=False):
    stochastic_tree = StochasticTree()
    stochastic_tree.fixed_state = fixed_state
    # add initial trees to open nodes
    for path_state, path_variant, task_tree in zip(path_states, path_variants, task_trees):
        startnode = STNode(
            next_id          = stochastic_tree.next_id,  # noqa
            depth            = 0,                        # noqa
            startidx         = 0,                        # noqa
            state            = state,                    # noqa
            path_state       = path_state,               # noqa
            path_variant     = path_variant,             # noqa
            task_tree        = task_tree,                # noqa
            possible_actions = possible_actions,         # noqa
            cumulative_cost  = 0.,                       # noqa
            cumulative_prob  = 1.,                       # noqa
        )
        stochastic_tree.nodes[startnode.id] = startnode
        stochastic_tree.root_nodes.append(startnode.id)
    if DEBUG:
        for _, node in stochastic_tree.nodes:
            print(node.task_tree)
    return stochastic_tree

def expand_stochastic_tree_node(stochastic_tree, nodeid, DEBUG=False):
    fixed_state = stochastic_tree.fixed_state
    node = stochastic_tree.nodes[nodeid]
    if not node.is_open:
        raise ValueError("Requested expansion on a closed node.")
    # Lazy update of the node state, get new task trees and options for the selected node
    is_changed = False
    new_path_state = node.path_state
    new_state = node.state
    if node.task_state_update is not None:
        if DYNAMIC_STATE:
            new_state = node.state.copy()
            is_changed = node.task_state_update.apply_to_state_along_taskpath(new_state)
            if is_changed:
                new_path = node.path_state.path_xy  # stays constant
                new_path_ij = node.path_state.path_ij  # stays constant
                new_path_state = path_state_from_state_and_path(new_state, new_path, new_path_ij)
        elif DYNAMIC_PATH_STATE:
            new_path_state = node.path_state.copy()
            is_changed = node.task_state_update.apply_to_path_state(new_path_state)
    # disable doing the same action again after failure
    less_actions = node.possible_actions
    if not ALLOW_CONSECUTIVE_FAILURE:
        raise NotImplementedError
    # if state has changed, recompute task tree, path_state
    new_path_variant = node.path_variant  # stays constant
    new_task_tree = node.task_tree
    if is_changed:
        task_length_limit = None
        if ACTION_LENGTH_LIMIT_IN_TASK_TREE is not None:
            task_length_limit = int(ACTION_LENGTH_LIMIT_IN_TASK_TREE / fixed_state.map.resolution())
        new_task_tree = task_tree_from_path_state(new_path_state, less_actions,
                                                  task_length_limit, constrain_to_depth_1=True)
    # if goal is reached
    is_terminal = False
    new_startidx = new_path_state.pos_along_path
    if new_startidx == len(new_path_state.path_xy) - 1:
        # create an empty task tree to make the node terminal
        new_task_tree = {new_startidx: TTNode()}
        is_terminal = True
    # update all stale fields
    node.state = new_state
    node.possible_actions = less_actions
    node.path_state = new_path_state
    node.path_variant = new_path_variant
    node.task_tree = new_task_tree
    node.startidx = new_startidx
    node.is_terminal = is_terminal

    # follow first branches, predict outcome, add resulting nodes to tree
    newnodes = []
    first_branches = node.task_tree[node.startidx].childbranches
    for branch in first_branches:
        resultidx, action = branch.target, branch.action
        task_state = CTaskState(node.path_state, node.startidx, resultidx, fixed_state.map.resolution())
        outcomes = action.predict_outcome(task_state, params=None)
        # get resulting state for each outcome
        branch_bundle = BranchBundle(meta={"tasktargetpathidx": resultidx,
                                           "taskaction": action,
                                           "taskfullpath": node.path_state.get_path_xy()})
        for outcome in outcomes:
            # add new node to opennodes and nodes
            # use parent values for stale values which get updated later
            newnode = STNode(
                 next_id          = stochastic_tree.next_id,                          # noqa
                 depth            = node.depth+1,                                     # noqa
                 startidx         = node.startidx,                            # stale # noqa
                 state            = node.state,                               # stale # noqa
                 path_state       = node.path_state,                          # stale # noqa
                 path_variant     = node.path_variant,                        # stale # noqa
                 task_tree        = node.task_tree,                           # stale # noqa
                 possible_actions = node.possible_actions,                    # stale # noqa
                 cumulative_cost  = node.cumulative_cost+outcome.cost,                # noqa
                 cumulative_prob  = node.cumulative_prob*outcome.probability,         # noqa
            )
            newnode.task_state_update = outcome.task_state_update
            stochastic_tree.nodes[newnode.id] = newnode
            newnodes.append(newnode)
            branch_bundle.branches.append(
                Branch(target=newnode.id, action=action,
                       meta={"success": outcome.success,
                             "cost": outcome.cost,
                             "prob": outcome.probability})
            )
        node.childbranchbundles.append(branch_bundle)
    if not node.is_terminal and not node.childbranchbundles:
        warnings.warn("IA: Non terminal node has no branches. Check task trees and preconditions.")
    node.is_open = False
    return newnodes

def build_stochastic_tree(stochastic_tree, DEBUG=False):
    # find open nodes and keep an ranked heap
    def noderank(node):
        # breadth/depth first search
        if node.depth == 0:
            return -np.inf  # always expand root nodes first
#             return np.random.rand()  # random ordering
        return node.depth  # breadth first search
#             return -node.depth  # depth first search
    open_nodes = []
    for nodeid in stochastic_tree.nodes:
        node = stochastic_tree.nodes[nodeid]
        if node.is_open:
            heappush(open_nodes, (noderank(node), node.id))
    # iteratively expand nodes
    itercount = 0
    while open_nodes:
        itercount += 1
        _, nodeid = heappop(open_nodes)
        newnodes = expand_stochastic_tree_node(stochastic_tree, nodeid, DEBUG=DEBUG)
        for newnode in newnodes:
            heappush(open_nodes, (noderank(newnode), newnode.id))
        # stopping criteria
        if itercount > 100:
            break

def monte_carlo_tree_search(stochastic_tree, n_trials=1000, max_depth=10, DEBUG=False, INTERRUPT=None):
    for i in range(n_trials):
        # start a trial
        path_trough_tree = []
        if not stochastic_tree.root_nodes:
            return
        firstnodeidx = stochastic_tree.root_nodes[np.random.randint(
            len(stochastic_tree.root_nodes)
        )]
        node = stochastic_tree.nodes[firstnodeidx]
        # step through the tree, sampling outcomes and expanding nodes until end or max depth
        while True:
            # allow being interrupted externally
            if INTERRUPT is not None:
                if INTERRUPT[0]:
                    return
            # if depth is reached, leave node open, unexpanded
            if node.depth >= max_depth:
                break
            # expand and close open nodes
            if node.is_open:
                expand_stochastic_tree_node(stochastic_tree, node.id, DEBUG=DEBUG)
            # termination conditions
            if node.is_terminal:
                break
            if not node.childbranchbundles:
#                 warnings.warn("Node {} is childless, non-terminal, and closed.".format(node.id))
                break
            # pick action at random
            bundleidx = np.random.randint(len(node.childbranchbundles))
            bundle = node.childbranchbundles[bundleidx]
            path_trough_tree.append((node, bundleidx))  # add action to the path
            # pick outcome stochastically
            choices = [branch.target for branch in bundle.branches]
            probs = np.array([branch.meta["prob"] for branch in bundle.branches], dtype=np.float32)
#                 resultid = np.random.choice(choices, p=probs)  # 10x slower
            resultid = choices[sultans_wife(probs)]
            # update for next loop
            node = stochastic_tree.nodes[resultid]
        # get final cost, add to MC estimate
        endnode = node
        final_is_goal_reached = node.is_terminal
        final_cost = node.cumulative_cost
        if not final_is_goal_reached:
            final_cost = np.nan
        # estimate for first choice, between three paths
        for node, bundleidx in path_trough_tree:
            # create NodeMCEstimateInfo for node if it doesn't have one
            if node.mce_info is None:
                node.mce_info = NodeMCEstimateInfo(
                    [MCBranchEstimate() for _ in node.childbranchbundles])
            estimate = node.mce_info.option_mc_estimates[bundleidx]
            estimate.n_total += 1
            if final_is_goal_reached:
                estimate.n_goal_reached += 1
                estimate.sum_cost_to_goal += final_cost
            if STORE_ESTIMATE_DISTRIBUTIONS:
                estimate.dist.append(final_cost)
        # add final node
        if endnode.mce_info is None:
            endnode.mce_info = NodeMCEstimateInfo(
                [MCBranchEstimate() for _ in endnode.childbranchbundles])
        estimate = endnode.mce_info.endnode_estimate
        estimate.n_total += 1
        if final_is_goal_reached:
            estimate.n_goal_reached += 1
            estimate.sum_cost_to_goal += final_cost
        if STORE_ESTIMATE_DISTRIBUTIONS:
            estimate.dist.append(final_cost)

def greedy_prune_tree(stochastic_tree, DEBUG=False):
    def objective_score(mc_estimate):
        # TODO : pick either lowest cost sequence or mix of lowest cost and low outcome variance
        expcost = mc_estimate.expected_final_cost()
        penalty = mc_estimate.n_total / mc_estimate.n_goal_reached
        return expcost * penalty

    def is_first_score_better(first_score, second_score):
        return first_score <= second_score

    def worst_possible_score():
        return np.inf
    # new pruned trees
    pruned_stochastic_tree = StochasticTree()
    pruned_stochastic_tree.fixed_state = stochastic_tree.fixed_state
    pruned_stochastic_tree.next_id = stochastic_tree.next_id
    # only add best options to pruned trees
    bestnodeid = None
    # find start node
    bestobjectivescore = worst_possible_score()
    for nodeid in stochastic_tree.root_nodes:
        mce_info = stochastic_tree.nodes[nodeid].mce_info
        if mce_info is None:
            continue
        est = mce_info.node_estimate()
        if est.n_goal_reached == 0:
            continue
        objectivescore = objective_score(est)
        if is_first_score_better(objectivescore, bestobjectivescore):
            bestobjectivescore = objectivescore
            bestnodeid = nodeid
    if bestnodeid is None:
        raise ValueError("No estimates found for root nodes")
    # add best root node to pruned trees
    pruned_stochastic_tree.root_nodes = [bestnodeid]
    # add root node to nodes_to_copy
    nodes_to_copy = [bestnodeid]
    # greedy step through tree
    while nodes_to_copy:
        # pick an unprocessed node, copy it into the tree
        nodeid = nodes_to_copy.pop()
        node = stochastic_tree.nodes[nodeid]
        pruned_stochastic_tree.nodes[nodeid] = node.copy()
        # Find best action for this node, add outcomes to nodes_to_copy
        if node.is_terminal or node.is_open:
            continue
        mce_info = node.mce_info
        if mce_info is None:
            print("ERROR: closed non terminal node with no mce_info")
            continue
        estimates = mce_info.option_mc_estimates
        bestobjectivescore = worst_possible_score()
        bestoptionidx = None
        for i, est in enumerate(estimates):
            if est.n_goal_reached == 0:
                continue
            objectivescore = objective_score(est)
            if is_first_score_better(objectivescore, bestobjectivescore):
                bestobjectivescore = objectivescore
                bestoptionidx = i
        if bestoptionidx is None and estimates:
            # there are branches but none of them lead to the goal. pick any explored one.
            for i, est in enumerate(estimates):
                if est.n_total > 0:
                    bestoptionidx = i
        if bestoptionidx is None:
            continue
        bestoptionbundle = node.childbranchbundles[bestoptionidx]
        # leave only best option in the stochastic tree
        pruned_stochastic_tree.nodes[nodeid].childbranchbundles = [bestoptionbundle]
        pruned_stochastic_tree.nodes[nodeid].mce_info.option_mc_estimates = [
            node.mce_info.option_mc_estimates[bestoptionidx]
        ]
        # add all option outcomes to nodes_to_copy
        bundle = bestoptionbundle
        for branch in bundle.branches:
            nodes_to_copy.append(branch.target)
    # result
    return pruned_stochastic_tree

def select_best_sequence_from_mcts_estimates(stochastic_tree, DEBUG=False):
    def objective_score(mc_estimate):
        # TODO : pick either lowest cost sequence or mix of lowest cost and low outcome variance
        return mc_estimate.expected_final_cost()

    def is_first_score_better(first_score, second_score):
        return first_score <= second_score

    def worst_possible_score():
        return np.inf
    sequence = []
    bestnodeid = None
    # find start node
    bestobjectivescore = worst_possible_score()
    for nodeid in stochastic_tree.root_nodes:
        mce_info = stochastic_tree.nodes[nodeid].mce_info
        if mce_info is None:
            continue  # we don't have monte carlo info on this node
        est = mce_info.node_estimate()
        if est.n_goal_reached == 0:
            continue
        objectivescore = objective_score(est)
        if is_first_score_better(objectivescore, bestobjectivescore):
            bestobjectivescore = objectivescore
            bestnodeid = nodeid
        if DEBUG:
            print("ROOT NODE {} ---------------".format(nodeid))
            print(est.n_goal_reached)
            print(est.n_total)
            print(est.sum_cost_to_goal)
            p = None
            objectivescore = None
            if est.n_total > 0:
                p = est.prob_reach_goal()
            if est.n_goal_reached > 0:
                objectivescore = objective_score(est)
            print("p: ", p)
            print("objectivescore: ", objectivescore)
    if bestnodeid is None:
        raise ValueError("No estimates found for root nodes")
    # greedy step through tree
    nodeid = bestnodeid
    while True:
        if DEBUG:
            print("RESULTING NODE: {} ---------------".format(nodeid))
        if stochastic_tree.nodes[nodeid].is_terminal:
            if DEBUG:
                print("GOAL REACHED")
            break
        estimates = stochastic_tree.nodes[nodeid].mce_info.option_mc_estimates
        bestobjectivescore = worst_possible_score()
        bestoptionidx = None
        for i, est in enumerate(estimates):
            if est.n_goal_reached == 0:
                continue
            objectivescore = objective_score(est)
            if is_first_score_better(objectivescore, bestobjectivescore):
                bestobjectivescore = objectivescore
                bestoptionidx = i
            if DEBUG:
                print("-- Option -- ")
                print(est.n_goal_reached)
                print(est.n_total)
                print(est.sum_cost_to_goal)
                p = None
                objectivescore = None
                if est.n_total > 0:
                    p = est.prob_reach_goal()
                if est.n_goal_reached > 0:
                    objectivescore = objective_score(est)
                print("p: ", p)
                print("objectivescore: ", objectivescore)
        if bestoptionidx is None:
            break
        if DEBUG:
            print("SELECTED OPTION -------> {}".format(bestoptionidx))
        sequence.append((nodeid, bestoptionidx))
        # follow option stochastically
        bundle = stochastic_tree.nodes[nodeid].childbranchbundles[bestoptionidx]
        choices = [branch.target for branch in bundle.branches]
        probs = [branch.meta["prob"] for branch in bundle.branches]
        nodeid = np.random.choice(choices, p=probs)
    # result
    return sequence

def optimistic_sequence_from_pruned_tree(pruned_stochastic_tree):
    sequence = []
    # start node
    nodeid = pruned_stochastic_tree.root_nodes[0]
    # step through tree, picking optimistic outcomes
    while True:
        if nodeid not in pruned_stochastic_tree.nodes:
            break
        if pruned_stochastic_tree.nodes[nodeid].is_terminal:
            break
        bestoptionidx = 0
        # follow option stochastically
        if not pruned_stochastic_tree.nodes[nodeid].childbranchbundles:
            break
        bundle = pruned_stochastic_tree.nodes[nodeid].childbranchbundles[bestoptionidx]
        successbranches = [branch for branch in bundle.branches if branch.meta["success"] == 1]
        successprobs = np.array([branch.meta["prob"] for branch in successbranches])
        successprobs = successprobs / np.sum(successprobs)  # normalize to sum = 1
        branch = np.random.choice(successbranches, p=successprobs)
        nodeid = branch.target
        sequence.append((bundle.meta, branch.meta))
    # result
    return sequence

def firstfailure_then_optimistic_sequence_from_pruned_tree(pruned_stochastic_tree):
    sequence = []
    # start node
    nodeid = pruned_stochastic_tree.root_nodes[0]
    # step through tree, picking optimistic outcomes
    first = True
    while True:
        if nodeid not in pruned_stochastic_tree.nodes:
            break
        if pruned_stochastic_tree.nodes[nodeid].is_terminal:
            break
        if not pruned_stochastic_tree.nodes[nodeid].childbranchbundles:
            break
        bestoptionidx = 0
        # follow option stochastically
        bundle = pruned_stochastic_tree.nodes[nodeid].childbranchbundles[bestoptionidx]
        if first:
            failurebranches = [branch for branch in bundle.branches if branch.meta["success"] < 1]
            failureprobs = np.array([branch.meta["prob"] for branch in failurebranches])
            failureprobs = failureprobs / np.sum(failureprobs)  # normalize to sum = 1
            branch = np.random.choice(failurebranches, p=failureprobs)
            first = False
        else:
            successbranches = [branch for branch in bundle.branches if branch.meta["success"] == 1]
            successprobs = np.array([branch.meta["prob"] for branch in successbranches])
            successprobs = successprobs / np.sum(successprobs)  # normalize to sum = 1
            branch = np.random.choice(successbranches, p=successprobs)
        nodeid = branch.target
        sequence.append((bundle.meta, branch.meta))
    # result
    return sequence

# UTILITY FUNCTIONS --------
def permutations_from_task_tree(task_tree):
    permutations = []

    def expandnode(node, task_tree, running_tasklist, permutations):
        # check if end reached
        child_branches = task_tree[node].childbranches
        if not child_branches:
            permutations.append(running_tasklist)
        for branch in child_branches:
            # prevent doing the same action over and over at same node (inf recursion)
            if node == branch.target:
                if branch.action in [task.action for task in running_tasklist]:
                    continue
            expandnode(branch.target, task_tree, running_tasklist + [branch, ], permutations)
    expandnode(0, task_tree, [], permutations)
    return permutations

# VISUALIZATION & OUTPUT ----
def visualize_task(action, path_ij):
    pass

def visualize_task_tree(task_tree):
    # assign value 1-N to each node in tree
    nodevals = {}
    for i, node in enumerate(task_tree):  # can also use sorted(task_tree)
        nodevals[node] = i
    # set of actions in tree
    actionset = set()
    for node in task_tree:
        branches = task_tree[node].childbranches
        for branch in branches:
            action = branch.action
            actionset.add(action)
    # assign float value 1-N to each action
    actionvals = {}
    for i, action in enumerate(actionset):
        actionvals[action] = i
    # plot tree
    legendcount = [0 for _ in actionset]  # hack to tie each action legend to a single branch
    for node in task_tree:
        node_x = nodevals[node]
        node_y = node
        plt.scatter([node_x], [node_y], color=(1, 1, 1, 1), edgecolors=(0, 0, 0, 1))
        branches = task_tree[node].childbranches
        for branch in branches:
            actionval = actionvals[branch.action]
            label = None
            if not legendcount[actionval]:
                label = branch.action.typestr()
                legendcount[actionval] = 1
            result_x = nodevals[branch.target]
            result_y = branch.target
            plt.plot([node_x, result_x], [node_y, result_y], ',-',
                     color=branch.action.color(), label=label)
    plt.gca().invert_yaxis()
    plt.gca().set_xticklabels([node for i, node in enumerate(task_tree)])
    plt.xticks([i for i, node in enumerate(task_tree)])
    plt.gca().set_yticklabels([node for i, node in enumerate(task_tree)])
    plt.yticks([node for i, node in enumerate(task_tree)])
    plt.legend()

def visualize_stochastic_tree(stochastic_tree, x_order="id"):
    # coordinate definitions
    if x_order == "id":
        def _x(node):
            return node.id
    elif x_order == "progress":
        def _x(node):
            return node.startidx
    elif x_order == "cost":
        def _x(node):
            return node.cumulative_cost
    elif x_order == "probability":
        def _x(node):
            return node.cumulative_prob
    else:
        raise NotImplementedError

    def _y(node):
        return node.depth
    # set of actions in tree
    actionset = set()
    for nodeid in stochastic_tree.nodes:
        bundles = stochastic_tree.nodes[nodeid].childbranchbundles
        for bundle in bundles:
            for branch in bundle.branches:
                actionset.add(branch.action)
    # assign float value 1-N to each action
    actionvals = {}
    for i, action in enumerate(actionset):
        actionvals[action] = i
    # plot tree
    legendcount = [0 for _ in actionset]  # hack to tie each action legend to a single branch
    for nodeid in stochastic_tree.nodes:
        node = stochastic_tree.nodes[nodeid]
        node_x = _x(node)
        node_y = _y(node)
        # draw nodes with no parents
        if node.depth == 0:
            plt.scatter([node_x], [node_y], color=(1, 1, 1, 1), edgecolors=(0, 0, 0, 1))
        # don't draw open node branches as they lead nowhere
        if node.is_open:
            continue
        # draw node branches
        bundles = node.childbranchbundles
        for bundle in bundles:
            branches = bundle.branches
            # find bundle midpoint
            mid_x = 0
            mid_y = 0
            for branch in branches:
                target_node = stochastic_tree.nodes[branch.target]
                result_x = _x(target_node)
                result_y = _y(target_node)
                mid_x += result_x
                mid_y += result_y
            mid_x = 1. * mid_x / len(branches)
            mid_y = 1. * mid_y / len(branches)
            fork_x = node_x + 0.5 * (mid_x - node_x)
            fork_y = node_y + 0.5 * (mid_y - node_y)
            # draw branches in bundle
            for branch in branches:
                target_node = stochastic_tree.nodes[branch.target]
                result_x = _x(target_node)
                result_y = _y(target_node)
                actionval = actionvals[branch.action]
                label = None
                if not legendcount[actionval]:
                    label = branch.action.typestr()
                    legendcount[actionval] = 1
                successcolor = "green" if branch.meta["success"] else "red"
                actioncolor = branch.action.color()
                # Y shaped branch
                plt.plot([node_x, fork_x, result_x], [node_y, fork_y, result_y], ',-',
                         color=actioncolor, label=label)
                if target_node.is_terminal:
                    plt.scatter([result_x], [result_y],
                                marker="*", color=(1, 1, 0, 1), edgecolors=(0.4, 0.4, 0, 1))
                else:
                    _ = "o" if target_node.is_open else "s"
                    plt.scatter([result_x], [result_y], color=(1, 1, 1, 1), edgecolors=successcolor)

    plt.gca().invert_yaxis()
    xticks = [nodeid for nodeid in stochastic_tree.nodes]
    if False:
        plt.gca().set_xticklabels(xticks)
        plt.xticks(xticks)
    # set of y ticks
    yticks = [stochastic_tree.nodes[nodeid].depth for nodeid in stochastic_tree.nodes]
    plt.gca().set_yticklabels(yticks)
    plt.yticks(yticks)
    plt.legend()
    plt.xlabel(x_order)
    plt.ylabel("depth")

def visualize_mc_estimate_tree(stochastic_tree, x_order="expcost", hide_unexplored=True):
    # coordinate definitions
    if x_order == "n_total":
        def _x(node):
            return node.mce_info.node_estimate().n_total
    elif x_order == "expcost":
        def _x(node):
            if node.mce_info is None:  # open node, not reached by MC
                # treat mce same as terminal node for visualization purposes
                return node.cumulative_cost
            x = node.mce_info.node_estimate().expected_final_cost()
            if np.isnan(x):
                x = node.cumulative_cost
            return x

    def _y(node):
        return node.depth
    # total trials
    max_width = 10
    n_total_trials = 0
    for nodeid in stochastic_tree.root_nodes:
        mce_info = stochastic_tree.nodes[nodeid].mce_info
        if mce_info is None:
            continue
        n_total_trials += mce_info.node_estimate().n_total
    # set of actions in tree
    actionset = set()
    for nodeid in stochastic_tree.nodes:
        bundles = stochastic_tree.nodes[nodeid].childbranchbundles
        for bundle in bundles:
            for branch in bundle.branches:
                actionset.add(branch.action)
    # set of path variants in tree
    pathvariantsset = set()
    for nodeid in stochastic_tree.root_nodes:
        pathvariantsset.add(stochastic_tree.nodes[nodeid].path_variant)
    # assign float value 1-N to each action
    actionvals = {}
    for i, action in enumerate(actionset):
        actionvals[action] = i
    # hack to tie each action legend to a single branch
    legendcount = [0 for _ in actionset]
    # assign success to each node
    for nodeid in stochastic_tree.nodes:
        stochastic_tree.nodes[nodeid].meta_success = None
        stochastic_tree.nodes[nodeid].meta_parent = None
    for nodeid in stochastic_tree.nodes:
        node = stochastic_tree.nodes[nodeid]
        bundles = node.childbranchbundles
        for i, bundle in enumerate(bundles):
            branches = bundle.branches
            for branch in branches:
                if branch.target not in stochastic_tree.nodes:
                    continue
                stochastic_tree.nodes[branch.target].meta_success = branch.meta["success"]
                stochastic_tree.nodes[branch.target].meta_parent = node.id
    # plot tree
    scatterpoints = {'x': [], 'y': [], 'c': [], 'edgecolors': []}
    terminalpoints = {'x': [], 'y': [], 'c': [], 'edgecolors': []}
    scatternodeidmap = {}
    # draw nodes
    for nodeid in stochastic_tree.nodes:
        node = stochastic_tree.nodes[nodeid]
        node_x = _x(node)
        node_y = _y(node)
        # node
        nodecolor = (1., 1., 1., 1.)  # white
        edgecolor = "green" if node.meta_success else "red"
        if node.depth == 0 or node.meta_success is None:
            edgecolor = "black"
        if node.mce_info is None:
            edgecolor = "grey"
            if hide_unexplored:
                continue
        if node.is_terminal:
            terminalpoints['x'].append(node_x)
            terminalpoints['y'].append(node_y)
            terminalpoints['c'].append((1., 1., 0., 1.))  # yellow
            terminalpoints['edgecolors'].append("black")
            nodecolor = (1., 1., 1., 1.)
            edgecolor = "white"
        scatterpoints['x'].append(node_x)
        scatterpoints['y'].append(node_y)
        scatterpoints['c'].append(nodecolor)
        scatterpoints['edgecolors'].append(edgecolor)
        scatternodeidmap[(node_x, node_y)] = node.id
    # draw node branches
    for nodeid in stochastic_tree.nodes:
        node = stochastic_tree.nodes[nodeid]
        node_x = _x(node)
        node_y = _y(node)
        bundles = node.childbranchbundles
        for i, bundle in enumerate(bundles):
            branches = bundle.branches
            # find bundle midpoint
            mid_x = 0
            mid_y = 0
            n_tines = 0  # fork tines
            for branch in branches:
                if branch.target not in stochastic_tree.nodes:
                    continue
                target_node = stochastic_tree.nodes[branch.target]
                if target_node.mce_info is None:
                    if hide_unexplored:
                        continue
                result_x = _x(target_node)
                result_y = _y(target_node)
                mid_x += result_x
                mid_y += result_y
                n_tines += 1
            fork_x = node_x
            fork_y = node_y
            if n_tines > 0:
                mid_x = 1. * mid_x / n_tines
                mid_y = 1. * mid_y / n_tines
                fork_x = node_x + 0.5 * (mid_x - node_x)
                fork_y = node_y + 0.5 * (mid_y - node_y)
            # draw branches in bundle
            for branch in branches:
                if branch.target not in stochastic_tree.nodes:
                    print("WARNING: stochastic tree has ghost branches (from {}, to {})".format(
                        node.id, branch.target))
                    plt.scatter(node_x, node_y, facecolors="black", marker="o", zorder=3)
                    print(node_x, node_y)
                    continue
                target_node = stochastic_tree.nodes[branch.target]
                result_x = _x(target_node)
                result_y = _y(target_node)
                actionval = actionvals[branch.action]
                label = None
                if not legendcount[actionval]:
                    label = branch.action.typestr()
                    legendcount[actionval] = 1
                actioncolor = branch.action.color()
                # Y shaped branch # linewidth varies with trials in bundle estimate
                # sqrt is used to make most comparable
                if target_node.mce_info is None:
                    if hide_unexplored:
                        continue
                    else:
                        linewidth = 0
                else:
                    linewidth = np.sqrt(
                        1. * target_node.mce_info.node_estimate().n_total / n_total_trials) * max_width
                linewidth = max(1, linewidth)
                linestyle = target_node.path_variant.linestyle()
                plt.plot([node_x, fork_x, result_x], [node_y, fork_y, result_y],
                         linestyle=linestyle, color=actioncolor,
                         label=label, linewidth=linewidth, zorder=1)
    # plot nodes
    sc = plt.scatter(zorder=2, marker="o", **scatterpoints)
    plt.scatter(zorder=2, marker="*", **terminalpoints)
    # Formatting
    plt.gca().invert_yaxis()
    xticks = [nodeid for nodeid in stochastic_tree.nodes]
    if False:
        plt.gca().set_xticklabels(xticks)
        plt.xticks(xticks)
    # set of y ticks
    yticks = [stochastic_tree.nodes[nodeid].depth for nodeid in stochastic_tree.nodes]
    plt.gca().set_yticklabels(yticks)
    plt.yticks(yticks)
    plt.legend()
    plt.xlabel(x_order)
    plt.ylabel("depth")
    plt.title("Monte Carlo estimates tree. {}".format(", ".join(["'{}' {}".format(
        variant.linestyle(), variant.typestr()) for variant in pathvariantsset])))
    # All this to have a live tooltip
    ax = plt.gca()
    fig = plt.gcf()
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        scindex = ind["ind"][0]
        pos = sc.get_offsets()[scindex]
        annot.xy = pos
        nodeindex = scatternodeidmap[(pos[0], pos[1])]
        node = stochastic_tree.nodes[nodeindex]
        children = [[branch.target for branch in bundle.branches] for bundle in node.childbranchbundles]
        notintree = [child for outcomes in children for child in outcomes
                     if child not in stochastic_tree.nodes]
        est = node.mce_info.node_estimate().expected_final_cost() if node.mce_info is not None else -1
        text = "node {}, depth {}, p {:.2f}, est {:.2f}, parent {}, children {}, n.i.t {}".format(
            node.id, node.depth, node.cumulative_prob, est, node.meta_parent, children, notintree)
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(scatterpoints['edgecolors'][scindex])
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
    # ---

def visualize_mc_estimate_distribution(estimate, n_bins=100, color=None, goal_not_reached_val=-1):
    if not STORE_ESTIMATE_DISTRIBUTIONS:
        raise NotImplementedError
    distribution = np.array(estimate.dist)
    distavg = np.mean(distribution[np.logical_not(np.isnan(distribution))])
    distribution[np.isnan(distribution)] = goal_not_reached_val
    plt.hist(distribution, bins=n_bins, color=color,
             histtype='stepfilled', alpha=0.3, density=True, ec="k")
    plt.axvline(distavg, color=color)
    plt.xlabel("expected cost")
    plt.ylabel("proportion of samples in bin")
    plt.title("Distribution of Expected Cost for Node")

def visualize_root_node_mc_estimate_distributions(stochastic_tree, n_bins=100, goal_not_reached_val=-1):
    legend = []
    for nodeid in stochastic_tree.root_nodes:
        node = stochastic_tree.nodes[nodeid]
        mce_info = node.mce_info
        if mce_info is None:
            continue
        visualize_mc_estimate_distribution(mce_info.node_estimate(),
                                           n_bins=n_bins,
                                           goal_not_reached_val=goal_not_reached_val,
                                           color=node.path_variant.linecolor())
        legend.append(node.path_variant.typestr())
    plt.legend(legend)
    plt.title("Distribution of Expected Cost for Root Nodes")

def visualize_task_permutations(task_permutations):
    # set of nodes
    nodeset = {0}
    for p in task_permutations:
        for task in p:
            nodeset.add(task.target)
    # set of actions in perm
    actionset = set()
    for p in task_permutations:
        for task in p:
            action = task.action
            actionset.add(action)
    # assign float value 1-N to each action
    actionvals = {}
    for i, action in enumerate(actionset):
        actionvals[action] = i
    legendcount = [0 for _ in actionset]
    for i, p in enumerate(task_permutations):
        prev = 0
        for task in p:
            actionval = actionvals[task.action]
            label = None
            if not legendcount[actionval]:
                label = task.action.typestr()
                legendcount[actionval] = 1
            color = task.action.color()
            plt.bar(i, [task.target - prev], bottom=[prev],
                    color=color, label=label,
                    edgecolor=(0, 0, 0, 1))
            prev = task.target
    plt.gca().invert_yaxis()
    plt.gca().set_xticklabels([])
    plt.xticks([])
    plt.gca().set_yticklabels([node for node in nodeset])
    plt.yticks([node for node in nodeset])
    plt.legend()

def visualize_action_preconditions(path_state, possible_actions):
    for i, action in enumerate(possible_actions):
        proto_segments, _ = action.check_preconditions_on_path(path_state)
        color = action.color()
        bool1d = bool1d_from_segments(proto_segments, len(path_state.get_path_xy()))
        plt.plot(bool1d, 's--', color=color)
    plt.legend([action.typestr() for action in possible_actions])

def visualize_dijkstra_field(state, fixed_state, path_variant):
    dijkstra = fixed_state.derived_state.dijkstra_from_goal(state, fixed_state, path_variant)
    gridshow(dijkstra)
    plt.title(path_variant.typestr() + "_dijkstra")

def visualize_dijkstra_fields(state, fixed_state, path_variants):
    n = len(path_variants)
    nrow = n
    ncol = 1
    fig, axes = plt.subplots(nrow, ncol)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for axrow, path_variant in zip(axes, path_variants):
        plt.sca(axrow)
        visualize_dijkstra_field(state, fixed_state, path_variant)

def visualize_traversability(state, fixed_state, path_variant):
    is_traversable = path_variant.is_state_traversable(state, fixed_state)
    gridshow(is_traversable.astype(int))
    plt.title(path_variant.typestr() + "_is_traversable")

def visualize_traversabilities(state, fixed_state, path_variants):
    n = len(path_variants)
    nrow = n
    ncol = 1
    fig, axes = plt.subplots(nrow, ncol)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for axrow, path_variant in zip(axes, path_variants):
        plt.sca(axrow)
        visualize_traversability(state, fixed_state, path_variant)

def visualize_state_feature(state, fixed_state, feature_name, hide_uncertain=True, uncertainties=False):
    if uncertainties:
        grid = 1. * state.grid_features_uncertainties()[kStateFeatures[feature_name], :]
    else:
        grid = 1. * state.grid_features_values()[kStateFeatures[feature_name], :]
    grid = np.copy(grid)
    if hide_uncertain and not uncertainties:
        grid[np.greater_equal(state.grid_features_uncertainties()[kStateFeatures[feature_name], :],
                              1.)] = np.nan
    grid[fixed_state.map.occupancy() > fixed_state.map.thresh_occupied()] = np.nan
    # color scale
    vmin = 0
    vmax = 1.
    if feature_name == "crowdedness":
        vmax = 6.
    if uncertainties:
        vmax = 2.
    gs = gridshow(grid, cmap=plt.cm.Greens, vmin=vmin, vmax=vmax)
    ij = state.get_pos_ij()
    plt.scatter(ij[0], ij[1], marker='+', c="black")
    uncertainty_str = " uncert." if uncertainties else ""
    plt.title(feature_name + uncertainty_str)
    return gs

def visualize_state_features(state, fixed_state, hide_uncertain=False):
    n = len(list(kStateFeatures))
    nrow = n
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol)
    for axrow, feature_name in zip(axes, kStateFeatures):
        plt.sca(axrow[0])
        visualize_state_feature(
            state, fixed_state, feature_name, hide_uncertain=hide_uncertain, uncertainties=False)
        plt.sca(axrow[1])
        visualize_state_feature(
            state, fixed_state, feature_name, hide_uncertain=hide_uncertain, uncertainties=True)

def visualize_world(state_e, fixed_state, possible_actions, env,
                    fig=None, possible_path_variants=None):
    paths_xy, path_variants = path_options(state_e, fixed_state,
                                           possible_path_variants=possible_path_variants)
    # ---------- PLOTTING
    if fig is None:
        fig = plt.figure("world")
        plt.cla()
    contours = fixed_state.map.as_closed_obst_vertices()
    fixed_state.map.plot_contours(contours, '-k')
    env.plot_agents_xy()
    for path_xy, variant in zip(paths_xy, path_variants):
        plt.plot(path_xy[:, 0], path_xy[:, 1], color=variant.linecolor())
    plt.axis('equal')

def visualize_simple(state_e, fixed_state, pruned_stochastic_tree, byproducts):
    # From byproducts
    path_variants = byproducts["path_variants"]
    path_states = byproducts["path_states"]
    # plot
    visualize_state_feature(state_e, fixed_state, "crowdedness", hide_uncertain=True)
    # paths
    for i, (path_state, path_variant) in enumerate(zip(
            path_states, path_variants)):
        path_ij = path_state.get_path_ij()
        plt.plot(path_ij[:, 0], path_ij[:, 1], '--', color=path_variant.linecolor())
    # chosen path
    if pruned_stochastic_tree.root_nodes:
        node = pruned_stochastic_tree.nodes[pruned_stochastic_tree.root_nodes[0]]
        chosen_path_variant = node.path_variant
        chosen_path_state = node.path_state
        path_ij = chosen_path_state.get_path_ij()
        plt.plot(path_ij[:, 0], path_ij[:, 1], color=chosen_path_variant.linecolor())

def visualize_sequence_as_path(optimistic_sequence, fixed_state):
    prev_idx = 0
    for task, cps in optimistic_sequence:
        target_idx = task["tasktargetpathidx"]
        action = task["taskaction"]
        path_ij = fixed_state.map.xy_to_floatij(task["taskfullpath"][prev_idx:target_idx])
        plt.plot(path_ij[:, 0], path_ij[:, 1], color=action.color())
        plt.scatter(path_ij[-1, 0], path_ij[-1, 1], color='k', marker='.', zorder=10)
        prev_idx = target_idx

def visualize_planning_process(state_e, fixed_state, possible_actions,
                               pruned_stochastic_tree, optimistic_sequence, firstfailure_sequence,
                               byproducts, env=None):
    # ----------------- PROCESSING -------------------------------------------------------------------
    # From byproducts
    path_variants = byproducts["path_variants"]
    path_states = byproducts["path_states"]
    task_trees = byproducts["task_trees"]
    stochastic_tree = byproducts["stochastic_tree"]
    # ---------------- PLOTTING ----------------------------------------------------------------------
    visualize_state_features(state_e, fixed_state, hide_uncertain=False)
    visualize_traversabilities(state_e, fixed_state, paths.PATH_VARIANTS)
    visualize_dijkstra_fields(state_e, fixed_state, paths.PATH_VARIANTS)
    if env is not None:
        visualize_world(state_e, fixed_state, possible_actions, env)
    # Path Options & Task Trees
    for i, (path_state, path_variant, tree) in enumerate(zip(
            path_states, path_variants, task_trees)):
        path_xy = path_state.get_path_xy()
        path_ij = path_state.get_path_ij()
        fig, axes = plt.subplots(3, 2, num=path_variant.typestr())
        plt.sca(axes[0, 0])
        plt.title("permissivity_map")
        visualize_state_feature(state_e, fixed_state, "permissivity")
        plt.plot(path_ij[:, 0], path_ij[:, 1], color=path_variant.linecolor())
        plt.sca(axes[0, 1])
        plt.title("path in world")
        contours = fixed_state.map.as_closed_obst_vertices()
        fixed_state.map.plot_contours(contours, '-k')
        if env is not None:
            env.plot_agents_xy()
        plt.plot(path_xy[:, 0], path_xy[:, 1], color=path_variant.linecolor())
        plt.axis('equal')
        plt.sca(axes[1, 0])
        plt.title("state features along path")
        for feature_name in kStateFeatures:
            n = kStateFeatures[feature_name]
            values = path_state.path_features_values()[n, :]
            uncertainties = path_state.path_features_uncertainties()[n, :]
            max_y = max(np.max(uncertainties), np.max(values))
            plt.plot(values, 's--', label=feature_name, alpha=0.5)
            plt.plot(uncertainties, 'o--', label=feature_name + "_uncertainty", alpha=0.5)
            plt.ylim([0, min(max_y, 5.)])
        plt.xlim([0, len(path_state.get_path_xy())])
        plt.legend()
        plt.sca(axes[1, 1])
        plt.title("task tree for path")
        visualize_task_tree(tree)
        plt.sca(axes[2, 0])
        plt.title("action preconditions along path")
        visualize_action_preconditions(path_state, possible_actions)  # SEGFAULT?
        plt.sca(axes[2, 1])
        plt.title("task permutations for path")
        permutations = permutations_from_task_tree(tree)
        visualize_task_permutations(permutations)
    plt.figure("mc_estimate_tree")
    visualize_mc_estimate_tree(stochastic_tree)
    plt.figure("pruned_tree")
    visualize_mc_estimate_tree(pruned_stochastic_tree)
    plt.figure("outcome_distributions")
    visualize_root_node_mc_estimate_distributions(stochastic_tree)
    plot_closeall_button()

def print_tree(task_tree):
    for node in task_tree:
        print("---------- {} ---------".format(node))
        branches = task_tree[node].childbranches
        print([b[0] for b in branches])

def print_mcts_estimates(stochastic_tree):
    for nodeid in stochastic_tree.root_nodes:
        mce_info = stochastic_tree.nodes[nodeid].mce_info
        print("ROOT NODE {} ---------------".format(nodeid))
        if mce_info is None:
            print("no MC info")
            continue
        est = mce_info.node_estimate()
        print(est.n_goal_reached)
        print(est.n_total)
        print(est.sum_cost_to_goal)
        p = None
        expcost = None
        if est.n_total > 0:
            p = est.prob_reach_goal()
        if est.n_goal_reached > 0:
            expcost = est.expected_final_cost()
        print("p: ", p)
        print("expcost: ", expcost)
    for nodeid in stochastic_tree.nodes:
        mce_info = stochastic_tree.nodes[nodeid].mce_info
        print("NODE {} ---------------".format(nodeid))
        if mce_info is None:
            print("no MC info")
            continue
        estimates = mce_info.option_mc_estimates
        for est in estimates:
            print("-- Action -- ")
            print(est.n_goal_reached)
            print(est.n_total)
            print(est.sum_cost_to_goal)
            p = None
            expcost = None
            if est.n_total > 0:
                p = est.prob_reach_goal()
            if est.n_goal_reached > 0:
                expcost = est.expected_final_cost()
            print("p: ", p)
            print("expcost: ", expcost)
