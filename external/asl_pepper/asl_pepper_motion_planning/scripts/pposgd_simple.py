from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

rew_hist_filepath = '/tmp/rew_hist.txt'

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    ob = env.reset()
    n_agents = len(ob[1])
    new = [True for _ in range(n_agents)] # marks if we're on first timestep of an episode

    # right now all agents get reset together if an agent crashes and the
    # episode ends. this means that episode length is the same for every agent
    cur_ep_ret = np.zeros(n_agents) # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment. shape ([dynamic], n_agents)
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.zeros((horizon, )+ob[0].shape, 'float32')
    obs_2 = np.zeros((horizon, )+ob[1].shape, 'float32')
    rews = np.zeros((horizon, n_agents), 'float32')
    vpreds = np.zeros((horizon, n_agents), 'float32')
    news = np.zeros((horizon, n_agents), 'int32')
    acs = np.zeros((horizon, n_agents)+ac.shape, 'float32')
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "ob_2" : obs_2, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i, :] = ob[0]
        obs_2[i, :] = ob[1]
        vpreds[i, :] = vpred
        news[i, :] = new
        acs[i, :] = ac
        prevacs[i, :] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        # what to do if an agent has crashed ? can't reset only that agent, 
        # otherwise other agents see a vanishing agent
        if np.any(new):
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret[:] = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew = seg["rew"]
    T = len(rew)
    seg["adv"] = gaelam = np.empty(rew.shape, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-np.any(new[t+1])
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        resume_training=False,
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(name="atarg", dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(name="ret", dtype=tf.float32, shape=[None]) # Empirical return

    summ_writer = tf.summary.FileWriter("/tmp/tensorboard", U.get_session().graph)
    U.launch_tensorboard_in_background("/tmp/tensorboard")

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule


    ob = U.get_placeholder_cached(name="ob")
    ob_2 = U.get_placeholder_cached(name="ob_2")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    check_ops = tf.add_check_numerics_ops()

    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    lossandgrad = U.function([ob, ob_2, ac, atarg, ret, lrmult],
            losses + [U.flatgrad(total_loss, var_list)],
            updates=[check_ops],
            )
    debugnan = U.function([ob, ob_2, ac, atarg, ret, lrmult],
            losses + [ratio, surr1, surr2])
    dbgnames = loss_names + ["ratio", "surr1", "surr2"]

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ob_2, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    if resume_training:
        pi.load_variables("/tmp/rlnav_model")
        oldpi.load_variables("/tmp/rlnav_model")
    else:
        # clear reward history log
        with open(rew_hist_filepath, 'w') as f:
            f.write('')

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ob_2, ac, atarg, tdlamret = seg["ob"], seg["ob_2"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        # flatten first two dimensions and put into batch maker
        dataset_dict = dict(ob=ob, ob_2=ob_2, ac=ac, atarg=atarg, vtarg=tdlamret)
        def flatten_horizon_and_agent_dims(array):
            """ using F order because we assume that the shape is (horizon, n_agents)
            and we want the new flattened first dimension to first run through
            horizon, then n_agents in order to not cut up the sequentiality of
            the experiences """
            new_shape = (array.shape[0] * array.shape[1], ) + array.shape[2:]
            return array.reshape(new_shape, order='F')
        for key in dataset_dict:
            dataset_dict[key] = flatten_horizon_and_agent_dims(dataset_dict[key])
        d = Dataset(dataset_dict, shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]
        n_agents = ob.shape[1]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        #export rewards to log file
        rewplot = np.array(seg["rew"])
        with open(rew_hist_filepath, 'ab') as f:
            np.savetxt(f, rewplot, delimiter=',')

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                landg = lossandgrad(batch["ob"], batch["ob_2"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                newlosses, g = landg[:-1], landg[-1]
                # debug nans
                if np.any(np.isnan(newlosses)):
                    dbglosses = debugnan(batch["ob"], batch["ob_2"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    raise ValueError("Nan detected in losses: {} {}".format(dbgnames, dbglosses))
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        pi.save_variables("/tmp/rlnav_model")
        pi.load_variables("/tmp/rlnav_model")

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ob_2"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", np.average([
            explained_variance(vpredbefore[:,i], tdlamret[:,i]) for i in
            range(n_agents)])) # average of explained variance for each agent
#         lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
#         listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
#         lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lens, rews = (seg["ep_lens"], seg["ep_rets"])
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        tf.summary.scalar("EpLenMean", np.mean(lenbuffer))
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def watch_rewplot(mode=0, path=rew_hist_filepath, pause_duration=1):
    from matplotlib import pyplot as plt
    try:
        first = True
        while True:
            try:
                rewplot = np.loadtxt(path, delimiter=",").T
                cumsum = np.cumsum(rewplot, axis=1)
            except:
                plt.clf()
                plt.pause(pause_duration)
                continue
            # reset sum at each crash / arrival
            final = rewplot * 1.
            final[np.abs(final) > 10] = -cumsum[np.abs(final) > 10]
            final_cumsum = np.cumsum(final, axis=1)
            # plot
            plt.ion()
            if mode == 0 or first: # Active
                fig = plt.gcf()
                fig.clear()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                lines1 = []
                lines2 = []
                for curve in rewplot:
                    line1, = ax1.plot(curve) # Returns a tuple of line objects, thus the comma
                    lines1.append(line1)
                for curve in cumsum:
                    line2, = ax2.plot(curve)
                    lines2.append(line2)
                plt.show()
            if mode == 1: # Passive
                for line1, curve in zip(lines1, rewplot):
                    line1.set_data(np.arange(len(curve)), curve)
                for line2, curve in zip(lines2, cumsum):
                    line2.set_data(np.arange(len(curve)), curve)
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(pause_duration)
            first = False
    except KeyboardInterrupt:
        mode += 1
        if mode > 1:
            plt.close()
            raise KeyboardInterrupt
        print("Switching to passive mode. Graph will still update, but panning and zoom is now free.")
        print("Ctrl-C again to close and quit")
        watch_rewplot(mode=1, path=path, pause_duration=pause_duration)

