from __future__ import print_function
from matplotlib.pyplot import imread
from functools import partial
import numpy as np
import os
import threading
from timeit import default_timer as timer
from yaml import load

from numba import njit
from math import sqrt, floor, log, ceil

@njit(fastmath=True, cache=True)
def compiled_compute_max_field(height, prev_field, result):
    height_prev = height-1
    o_prev = 2**height_prev # offset to previous field
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            max_val = 0
            # equivalent i j coordinates in the previous map to correct for padding
            # i_previous = i - o_prev
            i_p = i - o_prev
            j_p = j - o_prev
            # coordinates for 4 samples in the prev_field
            i_p_samples = [i_p, i_p + o_prev, i_p         , i_p + o_prev]
            j_p_samples = [j_p, j_p         , j_p + o_prev, j_p + o_prev]
            for n in range(4):
                i_p_ = i_p_samples[n]
                j_p_ = j_p_samples[n]
                if i_p_ < 0 or j_p_ < 0 or i_p_ >= prev_field.shape[0] or j_p_ >= prev_field.shape[1]:
                    continue
                val = prev_field[i_p_, j_p_]
                max_val = val if val > max_val else max_val
            result[i,j] = max_val

@njit(fastmath=True, cache=True)
def compiled_score(hits, field, node_ij, offset, h):
    score = 0
    o_f = 2**h - 1  # max_field offset
    o_w = offset # window offset
    node_i = node_ij[0]
    node_j = node_ij[1]
    fs_x = field.shape[0]
    fs_y = field.shape[1]
    for n in range(len(hits)):
        ij = hits[n]
        i = ij[0]
        j = ij[1]
        # move hits to node_pos in window frame, 
        # then coordinates from window frame to field frame
        i_f = (i + node_i - o_w + o_f)
        j_f = (j + node_j - o_w + o_f)
        if i_f < 0 or j_f < 0 or i_f >= fs_x or j_f >= fs_y:
            continue
        score += field[i_f, j_f]
    return score



class Node(object):
    # node = (angle_index, height, pos_ij)
    def __init__(self, height=None, angle_index=None, pos_ij=None, score=None):
        self.h = height
        self.a = angle_index
        self.ij = np.array(pos_ij)
        self.score = score

class BranchAndBound(object):
    def __init__(self, reference_map, rot_downsampling=1.):
        """
        reference_map: the map to match against
        rot_downsampling: increase this value to reduce the amount of rotations considered
        """
        self.reference_map = reference_map
        self.precomputed_max_fields = [reference_map.occupancy(),]
        self.kRotDS = rot_downsampling

    def branch_and_bound(self, map_, match_threshold=0, theta_prior=0):
        problem_def = self._init_problem(map_)
        best_leaf_node = None
        best_score = match_threshold
        if match_threshold == '1/2 matched points':
            best_score = len(problem_def['rotated_hits'][0]) * 0.5
        print("Score threshold: {}".format(best_score))
        node_list = []

        tic = timer()
        # expand all rotation nodes
        for n in range(problem_def['n_rotations']):
            rot_node = Node(problem_def['init_height'] + 1, n, [0,0])
            node_list += self._expand_node(rot_node, problem_def) # get 4 children
        node_list.sort(key=lambda node: node.score)
        if theta_prior != 0:
            print("Sorting nodes according to distance from theta_prior ({} rad).".format(theta_prior))
            node_list.sort(key=lambda node: node.a * problem_def['dr'] - theta_prior, reverse=True)
        toc = timer()
        print("Expanding rotation nodes: {} s".format(abs(toc-tic)))

        tic = timer()
        # Depth First Search Greedy
        while node_list:
            node = node_list.pop()
            if node.score < best_score: # unpromising node
                continue
            if node.h == 0: # leaf node
                best_score = node.score
                best_leaf_node = node
                print("New best score: {}".format(best_score))
                continue
            # otherwise, promising inner node -> expand
            node_list += self._expand_node(node, problem_def)
        toc = timer()
        print("DFS greedy: {} s".format(abs(toc-tic)))

        # Resulting transform from map_ to reference
        if best_leaf_node is None:
            pose_ij = None
            th = None
        else:
            # TODO: compute actual pose
            pose_ij = best_leaf_node.ij - problem_def['window_offset']
            th = problem_def['dr'] * best_leaf_node.a
        return best_score, pose_ij, th


    def _init_problem(self, map_):
        tic = timer()
        problem_def = {}
        # window is reference_map + padding of half of map_ size on every side
        # this ensures at least a quarter, often > half of
        # the map_ intersects with the reference map
        half_length = int(max(map_.occupancy().shape[0], map_.occupancy().shape[1]) / 2.)
        window_limits = np.array([ # in occupancy frame
            [-half_length, self.reference_map.occupancy().shape[0]+half_length], # i_min, i_max
            [-half_length, self.reference_map.occupancy().shape[1]+half_length]]) # j_min, j_max
        window_size = window_limits[:,1] - window_limits[:,0]
        problem_def['window_offset'] = half_length
        problem_def['window_limits'] = window_limits
        problem_def['window_size'] = window_size
        # height of the first children nodes
        problem_def['init_height'] = int(ceil(log(max(window_size[1], window_size[0]), 2)) - 1)
        self.precompute_max_fields(problem_def['init_height'])
        # rotational resolution
        problem_def['dr'] = self.kRotDS / sqrt(map_.occupancy().shape[0]**2 + map_.occupancy().shape[1]**2)
        problem_def['n_rotations'] = int((2*np.pi / problem_def['dr']) - 1)
        # rotate map accordingly
        occupied_points = map_.as_occupied_points_ij()
        rotated_hits = [None] * problem_def['n_rotations']
        rotation_angles = [None] * problem_def['n_rotations']
        for n in range(problem_def['n_rotations']):
            th = problem_def['dr'] * n
            rot_mat = np.array([
                [np.cos(th), -np.sin(th)],
                [np.sin(th),  np.cos(th)],])
            # rotation around center
            rotated_hits[n] = (np.matmul(
                    rot_mat, 
                    (occupied_points - (np.array(map_.occupancy().shape) * 0.5)).T
                    ).T + (np.array(map_.occupancy().shape) * 0.5)).astype(int)
            rotation_angles[n] = th
        problem_def['rotated_hits'] = rotated_hits
        problem_def['rotation_angles'] = rotation_angles
        toc = timer()
        print("Problem initialization took {} s".format(abs(toc-tic)))
        print("  {} search window".format(problem_def['window_size']))
        print("  {} rotations".format(problem_def['n_rotations']))
        print("  total: {} leaf nodes".format(
            np.prod(problem_def['window_size']) * problem_def['n_rotations']))
        print("  precomputed max fields for windows of size:")
        print("  ", end="")
        print([2**h for h in range(len(self.precomputed_max_fields))])
        return problem_def

    def precompute_max_fields(self, up_to_height):
        n_new_fields_to_compute = up_to_height + 1 - len(self.precomputed_max_fields)
        for i in range(n_new_fields_to_compute):
            self.precomputed_max_fields.append(None)
        for i in range(1, len(self.precomputed_max_fields)):
            if self.precomputed_max_fields[i] is not None:
                continue
            height = i
            prev_field = self.precomputed_max_fields[i-1]
            result = np.zeros(np.array(self.reference_map.occupancy().shape) + 2**height - 1) # w/ necessary pad
            compiled_compute_max_field(height, prev_field, result)
            self.precomputed_max_fields[i] = result
            assert not np.isnan(result).any()
            assert not np.max(result) > 10e10

    def _expand_node(self, node, problem_def):
        # Split the node window into 4, of size 2^(node_height - 1)
        # if the window size is not a nice power of 2,
        # some children end up covering more of the window
        kChildrenOffsets = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],])
        child_nodes = []
        for offset in kChildrenOffsets:
            # ij are in window_ij frame
            child_h = node.h - 1
            child_ij = node.ij + ( 2**child_h * offset )
            if np.any(child_ij >= problem_def['window_size']):
                continue
            child_node = Node(child_h, node.a, child_ij)
            child_node.score = compiled_score(
                    problem_def['rotated_hits'][child_node.a],
                    self.precomputed_max_fields[child_node.h],
                    child_node.ij,
                    problem_def['window_offset'],
                    child_node.h)
            child_nodes.append(child_node)
        child_nodes.sort(key=lambda node: node.score) # sort by score
        return child_nodes


    def rotate_points_around_map_center(self, points, th, map_):
        rot_mat = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)],])
        # rotation around center
        return (np.matmul(
                rot_mat, 
                (points - (np.array(map_.occupancy().shape) * 0.5)).T
                ).T + (np.array(map_.occupancy().shape) * 0.5)).astype(int)

