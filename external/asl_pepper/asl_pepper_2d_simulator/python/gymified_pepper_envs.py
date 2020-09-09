from collections import OrderedDict
import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnv
from stable_baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

def evaluate_model(model, env, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  obs = env.reset()
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, dones, info = env.step(action)
      # Stats
      episode_rewards[-1] += rewards[0]
      if dones[0]:
          obs = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
  return mean_100ep_reward

from pepper_2d_simulator import PepperRLEnv
class PepperRLVecEnv(VecEnv):
    def __init__(self, args):
        self.env = PepperRLEnv(args)
        # Assumes merged lidar
        # no direct obstacle positions
        from gym.spaces.box import Box
        self.observation_space = Box(
            low=-100., high=100.,
            shape=(self.env.kObsBufferSize, self.env.kStateSize + self.env.kMergedScanSize),
            dtype=np.float32,
        )
        VecEnv.__init__(self, self.env.n_agents(), self.observation_space, self.env.action_space)
        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = self.env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs_t, rews, news, infos = self.env.step(self.actions)
        obss = self._obs_from_obs_tuple(obs_t)
        if np.any(news):
            reset_obs_t = self.env.reset(news)
            reset_obss = self._obs_from_obs_tuple(reset_obs_t)
        for env_idx in range(self.num_envs):
            obs = obss[env_idx]
            self.buf_rews[env_idx] = rews[env_idx]
            self.buf_dones[env_idx] = news[env_idx]
            self.buf_infos[env_idx] = infos[env_idx]
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then set reset observation
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs_t = self.env.reset()
                obs = reset_obss[env_idx]
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def seed(self):
        raise NotImplementedError

    def reset(self):
        obs_t = self.env.reset()
        obss = self._obs_from_obs_tuple(obs_t)
        for env_idx in range(self.num_envs):
            obs = obss[env_idx]
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        self.env.close()

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_obs_tuple(self, obs_t):
        for env_idx in range(self.num_envs):
            obss = np.zeros((self.num_envs, self.observation_space.shape))
            obss[env_idx,:,self.env.kStateSize:] = obs_t[0][env_idx,:,:,0] # lidar
            obss[env_idx,0,:self.env.kStateSize] = obs_t[1][env_idx]
        return obss

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

class PepperRLVecEnvRVO(VecEnv):
    def __init__(self, args):
        self.envs = [None for _ in range(args.n_envs)]
        map2d = None
        tsdf = None
        for env_idx in range(args.n_envs):
            self.envs[env_idx] = PepperRLEnv(args, map_=map2d, tsdf_=tsdf)
            map2d = self.envs[env_idx].map2d
            tsdf = self.envs[env_idx].tsdf
        # Assumes merged lidar
        # no direct obstacle positions
        from gym.spaces.box import Box
        self.observation_space = Box(
            low=-100., high=100.,
            shape=(self.envs[0].kObsBufferSize, self.envs[0].kStateSize + self.envs[0].kMergedScanSize),
            dtype=np.float32,
        )
        VecEnv.__init__(self, len(self.envs), self.observation_space, self.envs[0].action_space)
        self.keys, shapes, dtypes = obs_space_info(self.observation_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = [env.metadata for env in self.envs]

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
#             env_actions = self.envs[env_idx].get_rvo_action()
            env_actions = self.envs[env_idx].get_linear_controller_action() # DBG
            env_actions[0] = self.actions[env_idx]
            obs_t, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(env_actions, ONLY_FOR_AGENT_0=True)
            obs = self._obs_from_obs_tuple(obs_t)
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs_t = self.envs[env_idx].reset(ONLY_FOR_AGENT_0=True)
                obs = self._obs_from_obs_tuple(obs_t)
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def seed(self):
        raise NotImplementedError

    def reset(self):
        for env_idx in range(self.num_envs):
            obs_t = self.envs[env_idx].reset(ONLY_FOR_AGENT_0=True)
            obs = self._obs_from_obs_tuple(obs_t)
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env_idx in range(self.num_envs):
            self.envs[env_idx].close()

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_obs_tuple(self, obs_t):
        obs = np.zeros(self.observation_space.shape)
        obs[:,self.envs[0].kStateSize:] = obs_t[0][:,:,0] # lidar
        obs[0,:self.envs[0].kStateSize] = obs_t[1]
        return obs

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

# TODO ------
class IARLVecEnv(VecEnv):
    def __init__(self, *args):
        self.env = MultiIARLEnv(*args)
        VecEnv.__init__(self, self.env.n_envs, self.env.observation_space, self.env.action_space)
        obs_space = self.env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([
            (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
            for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = self.env.metadata

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]['terminal_observation'] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

