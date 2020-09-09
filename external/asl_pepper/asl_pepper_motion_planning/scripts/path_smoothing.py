from __future__ import print_function
import copy
import numpy as np
import os

from numba import njit
from math import sqrt, floor, log

# Working
def path_smoothing(path, local_comfort_tsdf, start_angle, end_angle, n_iterations=1000):
    new_path = copy.deepcopy(path)
    out_of_bounds = ( ( new_path[:,0].astype(int) >= local_comfort_tsdf.shape[0] ) +
                      ( new_path[:,0] < 0 ) +
                      ( new_path[:,1].astype(int) >= local_comfort_tsdf.shape[1] ) +
                      ( new_path[:,1] < 0 ) ) > 0
    if np.any(out_of_bounds):
        print(path)
        print("Path given to path_smoothing has out_of_bounds value")
        return [], []
    ## Inits
    start_anchor = path[0] - np.array([np.cos(start_angle), np.sin(start_angle)])
    start_anchor = start_anchor.reshape((1,2))
    end_anchor = path[-1] + np.array([np.cos(end_angle), np.sin(end_angle)])
    end_anchor = end_anchor.reshape((1,2))
    zero = np.array([0.])
    zeros = np.array([[0.,0.]])
    fals = np.array([False])
    freedom = np.zeros((len(path)))
    lfreedom = np.zeros((len(path)))
    rfreedom = np.zeros((len(path)))
    strain = np.zeros((len(path)))
    lpath = np.concatenate((start_anchor, path[:-1]))
    rpath = np.concatenate((path[1:], end_anchor))
    lcurv = np.zeros((len(path)))
    rcurv = np.zeros((len(path)))
    l_is_immobile = np.zeros((len(path))) > 0
    r_is_immobile = np.zeros((len(path))) > 0
    ldeltas = np.zeros_like(path)
    rdeltas = np.zeros_like(path)
    mask = np.ones((len(path))) > 0
    mask[0] = False
    mask[-1] = False
    for i in range(n_iterations):
        changed = np.where(mask)[0]
        lpath[1:] = path[:-1]
        rpath[:-1] = path[1:]

        # here curvature is a linear measure of distance from one node to the line drawn by
        # its two neighbor nodes:
        #
        #  A---B
        #  |\ /
        #  | x
        #  |/
        #  C 
        #
        # the curvature at A is the cross product of xA and CB (of norm = ||xA|| * ||CB|| * cos(th))
        # divided by the norm of CB to yield ||xA|| * cos(th). which is therefore proportional
        # to ||xA||. This is desired as it simplifies the calculation of point displacements.
        # The result is further divided by ||CB|| in order to render it scale invariant.
        midpoints = 0.5 * ( rpath + lpath )
        deltas = midpoints - path
        maj_segments = rpath - lpath
        cross = deltas[:,0] * maj_segments[:,1] - deltas[:,1] * maj_segments[:,0]
        curvature = cross / (maj_segments[:,0]**2 + maj_segments[:,1]**2) # norm is squared

        lcurv[1:] = curvature[:-1]
        rcurv[:-1] = curvature[1:]

        for n in changed:
            out_of_bounds = ( ( new_path[:,0].astype(int) >= local_comfort_tsdf.shape[0] ) +
                              ( new_path[:,0] < 0 ) +
                              ( new_path[:,1].astype(int) >= local_comfort_tsdf.shape[1] ) +
                              ( new_path[:,1] < 0 ) ) > 0
            if np.any(out_of_bounds):
                print(int(new_path[n,0]))
                print(n)
                print(new_path)
                print(path)
                print(out_of_bounds)
                print(i)
            val = local_comfort_tsdf[int(new_path[n,0]), int(new_path[n,1])]
            freedom[n] = val
            lfreedom[n+1] = val
            rfreedom[n-1] = val
        # heuristic: points which are the deepest inside the obstacle field should not move
        immobile = np.logical_and(lfreedom > freedom, freedom <= rfreedom)
        immobile[0] = True
        immobile[-1] = True
        l_is_immobile[1:] = immobile[:-1]
        r_is_immobile[:-1] = immobile[1:]

        ldeltas[1:] = deltas[:-1]
        rdeltas[:-1] = deltas[1:]
        # self smoothing (bring self to level of neighbors)
        if i % 3 == 0:
            posmin_neighbor_curvature = np.maximum(np.maximum(rcurv, 0), np.maximum(lcurv, 0))
            mask = curvature > posmin_neighbor_curvature
            strain = ((curvature - posmin_neighbor_curvature)/(curvature))
        elif i % 3 == 1:
            negmin_neighbor_curvature = -np.maximum(np.maximum(-rcurv, 0), np.maximum(-lcurv, 0))
            mask = curvature < negmin_neighbor_curvature
            strain = ((curvature - negmin_neighbor_curvature)/(curvature))
        # neighbor smoothing
        elif i % 11 == 0:
            mask = np.abs(curvature) < np.abs(lcurv)
            deltas[mask] = -ldeltas[mask]
            strain[:] = 0.5 # TODO actually equalize curvature
        elif i % 13 == 0:
            mask = np.abs(curvature) < np.abs(rcurv)
            deltas[mask] = -rdeltas[mask]
            strain[:] = 0.5 # TODO actually equalize curvature
        # immobile neighbor smoothing
        elif i % 97 == 0:
            mask = l_is_immobile
            deltas[mask] = -ldeltas[mask]
            strain[:] = 0.5 # TODO actually equalize curvature
        elif i % 101 == 0:
            deltas[mask] = -rdeltas[mask]
            strain[:] = 0.5 # TODO actually equalize curvature
        mask[immobile] = False
        new_path[mask] =  (deltas * strain.reshape((-1,1)) + path)[mask]
        out_of_bounds = ( ( new_path[:,0].astype(int) >= local_comfort_tsdf.shape[0] ) +
                          ( new_path[:,0] < 0 ) +
                          ( new_path[:,1].astype(int) >= local_comfort_tsdf.shape[1] ) +
                          ( new_path[:,1] < 0 ) ) > 0
        if np.allclose(path, new_path) and np.any(out_of_bounds):
            print("Goodbye cruel world")
            return [], []
        new_path[out_of_bounds] = path[out_of_bounds]
        path = new_path
    return path, immobile



def curvature_(path, start_angle, end_angle):
    """
    this measure of curvature is the inverse curvature radius
    of the curve going from the previous point to the next
    with the curve tangents at the previous and next point passing through the middle point.
    A----B
         |
         |
    x    C
    e.g. the curvature at B is defined as the inverse radius of the circle arc which passes through
    A and C (not B), with AB being the circle tangent at A and BC the circle tangent at C.
    (i.e. radius = xC)
    """
    start_anchor = path[0] - np.array([np.cos(start_angle), np.sin(start_angle)])
    end_anchor = path[-1] + np.array([np.cos(end_angle), np.sin(end_angle)])
    lpath = np.roll(path, 1, axis=0)
    lpath[0] = start_anchor
    rpath = np.roll(path, -1, axis=0)
    rpath[-1] = end_anchor
    segments = np.diff(path, axis = 0)
    angles = np.arctan2(segments[:,1], segments[:,0])
    angles = np.pad(angles, (1,1), mode='constant', constant_values=[start_angle,end_angle])
    dtheta = np.diff(angles, axis = 0)
    rinv = dtheta / np.linalg.norm(rpath - lpath, axis=-1)
    return rinv
