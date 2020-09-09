import numpy as np

from pyniel.numpy_tools import indexing

from cia import kStateFeatures as kSF

WARMUP_INITIAL_RADIUS = 1. # warmup means the path conditions are relaxed for the first few meters
class Path(object):
    def type(self):
        return type(self)

    def typestr(self):
        return str(self.type()).replace("<class 'pyIA.paths.", '').replace("'>", '')

    def letterstr(self):
        return self.typestr()[0]

    def linestyle(self):
        return '-'
    def linecolor(self):
        return 'black'

class NaivePath(Path):
    def linestyle(self):
        return '-'
    def linecolor(self):
        return 'tab:blue'
    def is_state_traversable(self, state, fixed_state):
        is_traversable = \
        fixed_state.map.occupancy() < fixed_state.map.thresh_occupied()
        return is_traversable
class PermissivePath(Path):
    def linestyle(self):
        return '-.'
    def linecolor(self):
        return 'orange'
    def is_state_traversable(self, state, fixed_state):
        is_traversable = \
        np.logical_and.reduce([
            np.logical_or( # high or unknown permissivity
                state.grid_features_values()[kSF["permissivity"]] >= 0.2, # high permissivity
                state.grid_features_uncertainties()[kSF["permissivity"]] >= 1., # low confidence
            ),
            np.logical_or( # low or unknown crowdedness
                state.grid_features_values()[kSF["crowdedness"]] <= 1., # low crowdedness
                state.grid_features_uncertainties()[kSF["crowdedness"]] >= 1., # low crowdedness
            ),
            fixed_state.map.occupancy() < fixed_state.map.thresh_occupied()
        ])
        return is_traversable
class StrictPermissivePath(Path):
    def linestyle(self):
        return '--'
    def linecolor(self):
        return 'tab:red'
    def is_state_traversable(self, state, fixed_state):
        is_traversable = \
        np.logical_and.reduce([
            np.logical_or( # very high or unknown permissivity
                state.grid_features_values()[kSF["permissivity"]] >= 0.8, # high permissivity
                state.grid_features_uncertainties()[kSF["permissivity"]] >= 1., # low confidence
            ),
            np.logical_or( # low or unknown crowdedness
                state.grid_features_values()[kSF["crowdedness"]] <= 1., # low crowdedness
                state.grid_features_uncertainties()[kSF["crowdedness"]] >= 1., # low crowdedness
            ),
            fixed_state.map.occupancy() < fixed_state.map.thresh_occupied()
        ])
        return is_traversable
class WarmupStrictPermissivePath(Path):
    def linestyle(self):
        return ':'
    def linecolor(self):
        return 'tab:purple'
    def is_state_traversable(self, state, fixed_state):
        sqr_initial_radius_ij = (WARMUP_INITIAL_RADIUS / fixed_state.map.resolution())**2
        ij = indexing.as_idx_array(fixed_state.map.occupancy(), axis='all')
        ii = ij[...,0]
        jj = ij[...,1]
        ai, aj = fixed_state.map.xy_to_floatij([state.get_pos()])[0][:2]
        is_in_initial_radius = ((ii - ai)**2 + (jj - aj)**2) < sqr_initial_radius_ij
        is_traversable = \
        np.logical_and(
            np.logical_or(
                np.logical_and(
                    np.logical_or( # very high or unknown permissivity
                        state.grid_features_values()[kSF["permissivity"]] >= 0.8, # high permissivity
                        state.grid_features_uncertainties()[kSF["permissivity"]] >= 1., # low confidence
                    ),
                    np.logical_or( # low or unknown crowdedness
                        state.grid_features_values()[kSF["crowdedness"]] <= 1., # low crowdedness
                        state.grid_features_uncertainties()[kSF["crowdedness"]] >= 1., # low crowdedness
                    )
                ),
                is_in_initial_radius
            ),
            fixed_state.map.occupancy() < fixed_state.map.thresh_occupied()
        )
        return is_traversable

PATH_VARIANTS = [NaivePath(), PermissivePath(), StrictPermissivePath(), WarmupStrictPermissivePath()]

def is_path_equivalent(a_path, b_path):
    if not np.allclose(np.shape(a_path), np.shape(b_path)):
        return False
    return np.allclose(a_path, b_path)

def is_path_in_list(path, list_):
    for b_path in list_:
        if is_path_equivalent(path, b_path):
            return True
    return False
