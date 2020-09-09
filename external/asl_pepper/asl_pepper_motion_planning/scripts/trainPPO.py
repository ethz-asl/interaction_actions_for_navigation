from __future__ import print_function
from socket import gethostname
import numpy as np
import os
import random
import tensorflow as tf
import gym
from PPO import *
from map2d import Map2D
from CMap2D import CMap2D
from pepper_2d_simulator import PepperRLEnv, populate_PepperRLEnv_args, check_PepperRLEnv_args
from pepper_2d_iarlenv import parse_training_args
from timeit import default_timer as timer

from smallMlp import smallMlp

def batchsplit(x, batchsize, axis=0):
    if axis != 0:
        raise NotImplementedError
    n_full_sections =  int(np.floor(len(x) / batchsize))
    if n_full_sections == 0:
        return [x]
    indices = [n * batchsize for n in range(1, n_full_sections+1)]
    sections = np.split(x, indices, axis=axis)
    if len(sections[-1]) == 0:
        sections = sections[:-1]
    return sections

def shuffle_batch(arrays):
    """ assumes input arrays are all of the form (BATCHSIZE, ...)
    with at least 2 dimensions
    """
    for i in range(len(arrays)):
        inds = np.arange(len(arrays[i]))
        np.random.shuffle(inds)
        break
    for i in range(len(arrays)):
        arrays[i] = arrays[i][inds,...]

def launch_tensorboard_in_background(log_dir):
    '''
    To log the Tensorflow graph when using rl-algs
    algorithms, you can run the following code
    in your main script:
        import threading, time
        def start_tensorboard(session):
            time.sleep(10) # Wait until graph is setup
            tb_path = osp.join(logger.get_dir(), 'tb')
            summary_writer = tf.summary.FileWriter(tb_path, graph=session.graph)
            summary_op = tf.summary.merge_all()
            launch_tensorboard_in_background(tb_path)
        session = tf.get_default_session()
        t = threading.Thread(target=start_tensorboard, args=([session]))
        t.start()
    '''
    import subprocess
    subprocess.Popen(['tensorboard', '--logdir', log_dir])


class Worker(object):
    def __init__(self, name, env, summary_writer, args, policy_type):
        self.scope = name
        self.env = env
        self.summary_writer = summary_writer
        self.ppo = policy_type(self.env.action_space, self.env.observation_space,self.scope, args)

        self.args = args
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.batch_size = args.batch_size
        self.optimization_iterations = args.optimization_iterations
        adam_actor = tf.train.AdamOptimizer(args.a_learning_rate, epsilon=1e-5)
        adam_critic = tf.train.AdamOptimizer(args.c_learning_rate, epsilon=1e-5)
        def apply_gradients_clipped(adamoptimizer, loss, max_grad_norm=self.args.clip_gradients):
            grads_and_var = adamoptimizer.compute_gradients(loss)
            grads, var = zip(*grads_and_var)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads_and_var = list(zip(grads, var))
            return adamoptimizer.apply_gradients(grads_and_var)
        self.train_op_actor = apply_gradients_clipped(adam_actor, self.ppo.actor_loss)
        self.train_op_critic = apply_gradients_clipped(adam_critic, self.ppo.critic_loss)
#         self.train_op_actor = tf.train.AdamOptimizer(args.a_learning_rate).minimize(self.ppo.actor_loss)
#         self.train_op_critic = tf.train.AdamOptimizer(args.c_learning_rate).minimize(self.ppo.critic_loss)



    def process(self, sess, saver, best_saver):
        """ Iterate over batches,
        [batch_size] steps (potentially several episodes) are collected into a batch
        it is then split into minibatches, each used for the optimization
        """

        self.summary_writer.add_graph(sess.graph)

        x_t = self.env.reset()
        x_t1 = 0
        n_agents = self.env.n_agents()
        episode_reward = np.zeros((n_agents,))
        episode_length = np.zeros((n_agents,))
        completed_episodes_rewards = []
        completed_episodes_lengths = []
        self.num_episodes = 0
        global_steps = sess.run(self.ppo.inc_global_steps)
        best_batch_reward = -1000000
        num_batches = 0

        # log initial state of variables
        summary = tf.Summary()
        summary.value.add(tag="Machine/{}".format(os.path.expandvars("$MACHINE_NAME")), simple_value=float(self.args.random_seed))
        summary.value.add(tag="HyperParams/Mode/{}".format(self.args.mode), simple_value=float(self.args.random_seed))
        if self.args.bounce_reset_vote:
            summary.value.add(tag="HyperParams/Mode/BounceResetVote", simple_value=float(self.args.random_seed))
        summary.value.add(tag="HyperParams/Mapname/{}".format(self.args.map_name), simple_value=float(self.args.random_seed))
        summary.value.add(tag="HyperParams/Deterministic", simple_value=float(self.args.deterministic))
        summary.value.add(tag="HyperParams/NAgents", simple_value=float(self.args.n_agents))
        summary.value.add(tag="HyperParams/dt", simple_value=float(self.args.dt))
        summary.value.add(tag="HyperParams/RewardCollision", simple_value=float(self.args.reward_collision))
        summary.value.add(tag="HyperParams/RewardArrival", simple_value=float(self.args.reward_arrival))
        summary.value.add(tag="HyperParams/RewardProgress", simple_value=float(self.args.reward_progress))
        summary.value.add(tag="HyperParams/RewardVelocity", simple_value=float(self.args.reward_velocity))
        summary.value.add(tag="HyperParams/RewardStandstill", simple_value=float(self.args.reward_standstill))
        summary.value.add(tag="HyperParams/RandomSeed", simple_value=float(self.args.random_seed))
        summary.value.add(tag="HyperParams/LearningRateActor", simple_value=float(self.args.a_learning_rate))
        summary.value.add(tag="HyperParams/LearningRateCritic", simple_value=float(self.args.c_learning_rate))
        summary.value.add(tag="HyperParams/EntropyCoefficient", simple_value=float(self.args.entropy_coefficient))
        summary.value.add(tag="HyperParams/MaxGradNorm", simple_value=float(self.args.clip_gradients))
        summary.value.add(tag="HyperParams/BatchSize", simple_value=float(self.args.batch_size))
        summary.value.add(tag="HyperParams/MinibatchSize", simple_value=float(self.args.minibatch_size))
        self.summary_writer.add_summary(summary, global_steps)
        self.summary_writer.flush()
        action, value = self.ppo.choose_action(x_t, sess)
        feed_dict = {}
        feed_dict[self.ppo.s_conv] = x_t[0]
        feed_dict[self.ppo.s] = x_t[1]
        if self.args.add_relative_obstacles:
            feed_dict[self.ppo.s_relobst] = x_t[2]
        feed_dict[self.ppo.target_returns] = np.zeros((action.shape[0], 1))
        feed_dict[self.ppo.is_training] = False
        feed_dict[self.ppo.entropy_coeff] = self.args.entropy_coefficient
        feed_dict[self.ppo.advantage] = np.zeros((action.shape[0], 1))
        feed_dict[self.ppo.a] = action
        merged_summaries = sess.run(self.ppo.merged_summaries, feed_dict=feed_dict)
        self.summary_writer.add_summary(merged_summaries, global_steps)
        self.summary_writer.flush()

        if self.args.progress_to_file:
            print("Host: {}".format(gethostname())
             , file=open(os.path.expanduser("~/{}.txt".format(self.args.run_folder)), "a+") )

        while True:
            if not os.path.exists(self.args.checkpoint_path):
                raise IOError("Model folder {} not found".format(self.args.checkpoint_path))
            num_batches += 1
            states_conv_buf = []
            states_buf = []
            if self.args.add_relative_obstacles:
                states_relobst_buf = []
            actions_buf = []
            rewards_buf = []
            values_buf = []
            terminal_buf = []

            batch_finished_episodes_rewards = []
            batch_finished_episodes_lengths = []
            batch_finished_episodes_end_step = []

            tic = timer()

            for i in range(0, self.batch_size):
                global_steps = sess.run(self.ppo.inc_global_steps)

                action, value = self.ppo.choose_action(x_t, sess)

                x_t1, r_t, terminal, info = self.env.step(action)

                terminal = np.logical_or(terminal, episode_length > self.args.max_episode_length) # stop after 400 steps

                episode_reward = episode_reward + np.array(r_t)
                episode_length = episode_length + 1
                states_conv_buf.append(x_t[0])
                states_buf.append(x_t[1])
                if self.args.add_relative_obstacles:
                    states_relobst_buf.append(x_t[2])
                actions_buf.append(action)
                values_buf.append(value)
                rewards_buf.append(r_t)
                terminal_buf.append(terminal)

                # If high frequency logging
                if False:
                    summary = tf.Summary()
                    for i, (r, re) in enumerate(zip(r_t, episode_reward)):
                        summary.value.add(tag='AgentSpecificRewards/Step_Reward_Agent_{}'.format(i), simple_value=float(r))
                        summary.value.add(tag='AgentSpecificRewards/Episode_Reward_Agent_{}'.format(i), simple_value=float(re))
                    self.summary_writer.add_summary(summary, global_steps)
                    self.summary_writer.flush()

                x_t = x_t1
                if np.any(terminal):
                    for i, is_terminal in enumerate(terminal):
                        if is_terminal:
                            self.num_episodes += 1
                            # log
                            EPISODE_SUMMARY = False # not v. informative, takes space if short episodes
                            if EPISODE_SUMMARY:
                                summary = tf.Summary()
                                summary.value.add(tag='Episodes/Reward', simple_value=float(episode_reward[i]))
                                summary.value.add(tag='Episodes/PerStep_Reward', simple_value=float(episode_reward[i]/episode_length[i]))
                                summary.value.add(tag='Episodes/Length', simple_value=float(episode_length[i]))
                                self.summary_writer.add_summary(summary, self.num_episodes)
                                self.summary_writer.flush()
                            # store
                            batch_finished_episodes_rewards.append(episode_reward[i])
                            batch_finished_episodes_lengths.append(episode_length[i])
                            batch_finished_episodes_end_step.append(global_steps)
                            # reset
                            episode_reward[i] = 0
                            episode_length[i] = 0
                    self.env.reset(terminal)

            batchavg_rew = np.mean(batch_finished_episodes_rewards)
            batchavg_len = np.mean(batch_finished_episodes_lengths)
            if batch_finished_episodes_lengths:
                batchmin_rew = np.min(batch_finished_episodes_rewards)
                batchmax_rew = np.max(batch_finished_episodes_rewards)
                batchmin_len = np.min(batch_finished_episodes_lengths)
                batchmax_len = np.max(batch_finished_episodes_lengths)
                summary = tf.Summary()
                summary.value.add(tag='Batches/Average_Episode_Reward', simple_value=float(batchavg_rew))
#                 summary.value.add(tag='Batches/Average_Episode_PerStep_Reward', simple_value=float(batchavg_rew/batchavg_len))
                summary.value.add(tag='Batches/Min_Episode_Reward', simple_value=float(batchmin_rew))
                summary.value.add(tag='Batches/Max_Episode_Reward', simple_value=float(batchmax_rew))
                summary.value.add(tag='Batches/Average_Episode_Length', simple_value=float(batchavg_len))
                summary.value.add(tag='Batches/Min_Episode_Length', simple_value=float(batchmin_len))
                summary.value.add(tag='Batches/Max_Episode_Length', simple_value=float(batchmax_len))
                self.summary_writer.add_summary(summary, global_steps)
                self.summary_writer.flush()

            toc = timer()
            batch_gen_time = toc-tic

            # log rewards
            print('ID :' + self.scope + ' - BATCH COMPLETE - total steps :' + str(
            global_steps)+ ', episodes finished :' + str(len(batch_finished_episodes_rewards))
             + ', average episode reward : {:.2f}'.format(batchavg_rew)
             + ', average episode length : {:.2f}'.format(batchavg_len)
             + ', batch generation time : {:.2f} sec'.format(batch_gen_time)
             )
            if self.args.progress_to_file:
                print('ID :' + self.scope + ' - BATCH COMPLETE - total steps :' + str(
                global_steps)+ ', episodes finished :' + str(len(batch_finished_episodes_rewards))
                 + ', average episode reward : {:.2f}'.format(batchavg_rew)
                 + ', average episode length : {:.2f}'.format(batchavg_len)
                 + ', batch generation time : {:.2f} sec'.format(batch_gen_time)
                 , file=open(os.path.expanduser("~/{}.txt".format(self.args.run_folder)), "a+") )

            tic = timer()


            # Save best model
            if batchavg_rew >= best_batch_reward:
                best_batch_reward = batchavg_rew
                self.ppo.save_best_model(sess, best_saver, global_steps)
                summary = tf.Summary()
                summary.value.add(tag='Batches/Overall_Best_Avg_Reward', simple_value=float(best_batch_reward))
                self.summary_writer.add_summary(summary, global_steps)
                self.summary_writer.flush()

            # Save model every n_steps
            save_every_n_steps = 131072 # 2^17
            save_every_n_batches = int(save_every_n_steps / self.batch_size)
            if save_every_n_batches < 1: save_every_n_batches = 1
            if num_batches % save_every_n_batches == 0 or num_batches == 1:
                self.ppo.save_model(sess, saver, global_steps)


            if len(states_buf) > 1:
                # peek into the next batch (value is only used if episode is not finished)
                next_value = self.ppo.get_value(x_t, sess)

                # General Advantage Estimate with Lambda
                returns = np.zeros_like(rewards_buf)
                values = np.array(values_buf).reshape(returns.shape)
                advantages = np.zeros_like(rewards_buf)
                nextgaelam = 0
                for t in reversed(range(len(rewards_buf))):
                    # true if the following step (t+1)  is part of the same episode as step (t)
                    tnonterminal = 1.0 - terminal_buf[t]
                    if t == len(rewards_buf) - 1:
                        nextvalues = next_value 
                    else:
                        nextvalues = values[t+1]
                    # delta are the 1-step advantages A^(1)
                    delta = rewards_buf[t] + self.gamma * nextvalues * tnonterminal - values[t]
                    # these are the inf-step advantages A^(1)_t + gam A^(1)_t+1 + gam**2 A^(1)_t+2 ...
                    # equivalent to A^(inf)_t = r_t + gam r_t+1 + ... - V(s_t) 
                    advantages[t] = nextgaelam = delta + self.gamma * self.lmbda * tnonterminal * nextgaelam
                    # old calculation for the returns
                returns = advantages + values
                if False:
                    discounted_r = np.zeros_like(rewards_buf)
                    for t in reversed(range(len(rewards_buf))):
                        tnonterminal = 1.0 - terminal_buf[t]
                        if t == len(rewards_buf) - 1:
                            nextdiscr = next_value
                        else:
                            nextdiscr = discounted_r[t+1]
                        # old calculation for the returns
                        discounted_r[t] = rewards_buf[t] + self.gamma * nextdiscr * tnonterminal
                    import matplotlib.pyplot as plt
                    plt.plot(np.array(rewards_buf)[:,0])
                    plt.plot(discounted_r[:,0])
                    plt.plot(returns[:,0])
                    plt.plot(values[:,0])
                    plt.plot(advantages[:,0])
                    plt.legend(['rew', 'dis', 'ret', 'val', 'adv'])
                    # add grid for terminal states
                    ax = plt.gca()
                    ax.set_xticks(np.where(np.array(terminal_buf)[:,0])[0], minor=False)
                    ax.set_xticks(np.where(np.abs(np.array(rewards_buf)[:,0]) > 10)[0], minor=True)
                    ax.xaxis.grid(True, which='major')
                    ax.xaxis.grid(True, which='minor')
                    ax.set_yticks([0], minor=False)
                    ax.yaxis.grid(True, which='major')
                    fd = {}
                    fd[self.ppo.s_conv] = np.array(states_conv_buf)[:,0]
                    fd[self.ppo.s] = np.array(states_buf)[:,0]
                    if self.args.add_relative_obstacles:
                        fd[self.ppo.s_relobst] = np.array(states_relobst_buf)[:,0]
                    fd[self.ppo.target_returns] = returns[:,:1] 
                    fd[self.ppo.is_training] = False
                    fd[self.ppo.entropy_coeff] = self.args.entropy_coefficient
                    fd[self.ppo.old_value] = values[:,:1]
                    fd[self.ppo.a] = np.array(actions_buf)[:,0]
                    fd[self.ppo.advantage] = advantages[:,:1]
                    cl, clcl = sess.run([self.ppo.critic_loss_, self.ppo.clipped_critic_loss],  feed_dict=fd)
                    plt.figure()
                    plt.plot(cl)
                    plt.plot(clcl)
                    plt.show()


                # turn buffers of shape 
                # (iterations, n_agents, ...) into
                # (iterations * n_agents, ...)
                def flatten_horizon_and_agent_dims(array):
                    """ using F order because we assume that the shape is (horizon, n_agents)
                    and we want the new flattened first dimension to first run through
                    horizon, then n_agents in order to not cut up the sequentiality of
                    the experiences """
                    new_shape = (array.shape[0] * array.shape[1], ) + array.shape[2:]
                    return array.reshape(new_shape, order='F')
                try:
                    if not self.args.add_relative_obstacles:
                        states_relobst_buf = np.zeros_like(states_buf) # dummy array
                    bsconv, bs, bsrelobst, ba, br, badv, bvold = (flatten_horizon_and_agent_dims(np.array(x)) 
                            for x in [states_conv_buf, states_buf, states_relobst_buf, actions_buf, returns, advantages, values])
                except IndexError as e:
                    print("IndexError")
                    print(e)
                    for x in [states_conv_buf, states_buf, actions_buf, returns, advantages, values]:
                        print(np.array(x).shape)
                    pass
                br = br[:,None]
                badv = badv[:,None]
                bvold = bvold[:,None]


                # pi -> old_pi
                sess.run(self.ppo.syn_old_pi)
                global_steps = sess.run(self.ppo.global_steps)

                # TODO use minibatch? shuffle?
                for _ in range(self.args.optimization_iterations):
                    # shuffle
                    shuffle_batch([bsconv, bs, bsrelobst, ba, br, badv, bvold])
                    # minibatch
                    minibatches = [minibatch for minibatch in zip(
                        *(batchsplit(b, self.args.minibatch_size) for b in [bsconv, bs, bsrelobst, ba, br, badv, bvold]))]

                    for mbsconv, mbs, mbsrelobst, mba, mbr, mbadv, mbvold in minibatches:
                        feed_dict_actor = {}
                        feed_dict_actor[self.ppo.s_conv] = mbsconv
                        feed_dict_actor[self.ppo.s] = mbs
                        if self.args.add_relative_obstacles:
                            feed_dict_actor[self.ppo.s_relobst] = mbsrelobst
                        feed_dict_actor[self.ppo.a] = mba
                        feed_dict_actor[self.ppo.advantage] = mbadv
                        feed_dict_actor[self.ppo.is_training] = True
                        feed_dict_actor[self.ppo.entropy_coeff] = self.args.entropy_coefficient
                        # unused, feed only for check numerics
                        feed_dict_actor[self.ppo.target_returns] = mbr # not used for computation, simply to feed placeholder

                        feed_dict_critic = {}
                        feed_dict_critic[self.ppo.s_conv] = mbsconv
                        feed_dict_critic[self.ppo.s] = mbs
                        if self.args.add_relative_obstacles:
                            feed_dict_critic[self.ppo.s_relobst] = mbsrelobst
                        feed_dict_critic[self.ppo.target_returns] = mbr
                        feed_dict_critic[self.ppo.is_training] = True
                        feed_dict_critic[self.ppo.entropy_coeff] = self.args.entropy_coefficient
                        feed_dict_critic[self.ppo.old_value] = mbvold
                        # unused, feed only for check numerics
                        feed_dict_critic[self.ppo.a] = mba
                        feed_dict_critic[self.ppo.advantage] = mbadv

                        sess.run([self.train_op_actor , self.ppo.batch_norm_update_op],  feed_dict=feed_dict_actor)
                        sess.run([self.train_op_critic, self.ppo.batch_norm_update_op], feed_dict=feed_dict_critic)

                # Log progressively less often
                summary_every_n_batches = 1
                if global_steps > 100000:
                    summary_every_n_batches = 10
                if num_batches % summary_every_n_batches == 0:
                    self.summary_log(sess, feed_dict_actor, feed_dict_critic, global_steps)

            toc = timer()
            optim_time = toc-tic
            print("Optim complete - time: {:.2f} s".format(optim_time))
            if self.args.progress_to_file:
                print("Optim complete - time: {:.2f} s".format(optim_time)
                 , file=open(os.path.expanduser("~/{}.txt".format(self.args.run_folder)), "a+") )

            if num_batches == 1 or num_batches % 10 == 0:
                summary = tf.Summary()
                summary.value.add(tag='Timings/BatchGenTime', simple_value=float(batch_gen_time))
                summary.value.add(tag='Timings/OptimTime', simple_value=float(optim_time))
                summary.value.add(tag='Timings/PerAgentStep_BatchGenTime', simple_value=float(batch_gen_time/(self.batch_size * self.args.n_agents)))
                summary.value.add(tag='Timings/PerAgentStep_OptimTime', simple_value=float(optim_time/(self.batch_size * self.args.n_agents)))
                self.summary_writer.add_summary(summary, global_steps)
                self.summary_writer.flush()






    def summary_log(self, sess, feed_dict_actor, feed_dict_critic, global_steps):
        """ This is where we log internal model data for diagnostics """

        # losses and internal summaries
        actor_loss, merged_summaries = sess.run([self.ppo.actor_loss, self.ppo.merged_summaries], feed_dict=feed_dict_actor)
        critic_loss = sess.run(self.ppo.critic_loss, feed_dict=feed_dict_critic)

        summary = tf.Summary()
        summary.value.add(tag='Losses/Actor_Loss', simple_value=float(actor_loss))
        summary.value.add(tag='Losses/Critic_Loss', simple_value=float(critic_loss))
        self.summary_writer.add_summary(summary, global_steps)
        self.summary_writer.add_summary(merged_summaries, global_steps)
        self.summary_writer.flush()




if __name__ == '__main__':
    tf.reset_default_graph()

    env_populate_args_func = populate_PepperRLEnv_args
    env_check_args_func = check_PepperRLEnv_args
    env_type = PepperRLEnv

    # args
    args, _ = parse_training_args(
      ignore_unknown=False,
      env_populate_args_func=env_populate_args_func
      )
    args.no_ros = True
    env_check_args_func(args)
    print(args)

    if args.policy == "MlpPolicy":
        policy_type = MlpPPO
    elif args.policy == "SmallMlpPolicy":
        policy_type = smallMlp
    else:
        raise NotImplementedError

    # summary
    summary_writer = tf.summary.FileWriter(args.summary_path)

    # random seed
    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Load environment
    env = env_type(args) # new PepperRLEnv(
    print("Created environment")

    # Run training
    chief =  Worker('Chief', env, summary_writer, args, policy_type)
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)
        best_saver = tf.train.Saver(max_to_keep=3, save_relative_paths=True)
        sess.run(tf.global_variables_initializer())
        if args.resume_from != '':
            chief.ppo.load_model(sess, saver)
        chief.process(sess, saver, best_saver)


