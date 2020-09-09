import tensorflow as tf
import numpy as np
import os
import  baselines.common.tf_util as U

# for versioning. Increment if changes make the model no longer backwards-compatible.
PPO_model_type_letter = "E"

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  tf.summary.histogram(var.name, var)
  with tf.name_scope('variables/{}'.format("".join([char if char not in ':/' else '_' for char in var.name ]))):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


class MlpPPO(object):
    def __init__(self, action_space, observation_space,scope, args):
        self.scope = scope
        self.EPSILON = 1e-8
        self.action_bound = [action_space.low, action_space.high]
        self.state_shape = observation_space[1].shape
        self.conv_state_shape = observation_space[0].shape
        DISCRETE = not args.continuous
        self.DIRECT_AGENT_OBS = len(observation_space) == 3
        if self.DIRECT_AGENT_OBS:
            self.relobst_state_shape = list(observation_space[2].shape)
            # set to fix sized of relative obstacles
            self.MAX_N_REL_OBSTACLES = args.max_n_relative_obstacles
            if self.relobst_state_shape[1] > self.MAX_N_REL_OBSTACLES:
                raise ValueError("Can only handle up to 10 dynamic obstacle states")
            self.relobst_state_shape[1] = self.MAX_N_REL_OBSTACLES
            self.relobst_state_shape = tuple(self.relobst_state_shape)
        if DISCRETE:
            assert len(action_space.shape) == 2
            self.num_action_values = action_space.shape[1]
        else:
            assert len(action_space.shape) == 1
        self.action_names = ['u', 'v', 'theta']
        self.num_action = action_space.shape[0]
        self.cliprange = args.cliprange
        self.checkpoint_path = args.checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.environment = args.environment

        self.global_steps = tf.Variable(0, dtype=tf.int32, name='global_steps', trainable=False)
        self.inc_global_steps = self.global_steps.assign_add(1)


        ## Build net
        with tf.variable_scope('input'):
            self.s_conv = tf.placeholder(name="s_conv", dtype="float", shape=(None, ) + self.conv_state_shape)
            self.s = tf.placeholder(name="s", dtype="float", shape=(None, ) + self.state_shape)
            if self.DIRECT_AGENT_OBS:
                self.s_relobst = tf.placeholder(name="s_relobst", dtype="float", shape=(None, ) + self.relobst_state_shape)
        with tf.variable_scope('action'):
            if DISCRETE:
                self.a = tf.placeholder(name="a", shape=[None, self.num_action], dtype=tf.float32)
            else:
                self.a = tf.placeholder(name="a", shape=[None, self.num_action], dtype=tf.float32)
        with tf.variable_scope('target_returns'):
            self.target_returns = tf.placeholder(name="target_returns", shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope('advantages'):
            self.advantage = tf.placeholder(name="advantage", shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope('is_training'):
            self.is_training = tf.placeholder(tf.bool, name="is_training")
        with tf.variable_scope('entropy_coeff'):
            self.entropy_coeff = tf.placeholder(name="entropy_coeff", dtype=tf.float32)
        with tf.variable_scope('old_predicted_values'):
            self.old_value = tf.placeholder(name="old_value", shape=[None, 1], dtype=tf.float32)

        assert len(self.state_shape) == 1
        assert self.state_shape[0] == 5
        state_relgoal, state_vel = tf.split(self.s, [2, 3], axis=1)
        if self.DIRECT_AGENT_OBS:
            state_relobst = U.flattenallbut0(self.s_relobst)
        features_relgoal = tf.nn.relu(tf.layers.dense(state_relgoal, 32, name='s_relgoal_preproc', kernel_initializer=U.normc_initializer(1.0)))
        features_vel = tf.nn.relu(tf.layers.dense(state_vel, 32, name='s_vel_preproc', kernel_initializer=U.normc_initializer(1.0)))
        if self.DIRECT_AGENT_OBS:
            features_relobst = tf.nn.relu(tf.layers.dense(state_relobst, 32, name='s_relobst_preproc', kernel_initializer=U.normc_initializer(1.0)))

        with tf.variable_scope('cnn'):
            with tf.variable_scope('lidar_preprocessing'):
                x = self.s_conv / 100. # from meters to adimensional
                # set x == 0 to x = 1
                x = 1 * tf.cast(tf.equal(x, 0), x.dtype) + x * tf.cast(tf.logical_not(tf.equal(x, 0)), x.dtype) # set 0 returns to 1 (max)
#                 x = tf.math.log(self.EPSILON + x) # inverse as close is more important than near

            kind = 'large'
            if kind == 'small': # from A3C paper
                x = tf.nn.relu(U.conv2d(x, 16, "l1", [4, 16], [1, 4], pad="SAME"))
                x = tf.nn.relu(U.conv2d(x, 32, "l2", [2, 4], [1, 2], pad="VALID"))
                x = U.flattenallbut0(x)
                x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            elif kind == 'large': # Nature DQN
                print(x.shape)
                x = tf.layers.batch_normalization(
                        tf.nn.relu(U.conv2d(x, 64, "l1", [1, 7], [1, 3], pad="SAME", summary_tag="Conv/Layer1")), training=self.is_training)
                print(x.shape)
                x = tf.layers.max_pooling2d(x, (1, 3), (1, 3), padding="SAME", name="Conv/MaxPool")
                xres = x
                print(x.shape)
                x = tf.layers.batch_normalization(
                        tf.nn.relu(U.conv2d(x, 64, "l2", [1, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer2")), training=self.is_training)
                print(x.shape)
                x = tf.layers.batch_normalization(
                        U.conv2d(x, 64, "l3", [1, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer3"), training=self.is_training)
                print(x.shape)
                xres2 = x
                x = tf.nn.relu(x + xres)
                x = tf.layers.batch_normalization(
                        tf.nn.relu(U.conv2d(x, 64, "l4", [2, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer4")), training=self.is_training)
                print(x.shape)
                x = tf.layers.batch_normalization(
                        U.conv2d(x, 64, "l5", [2, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer5"), training=self.is_training)
                print(x.shape)
                x = tf.nn.relu(x + xres2)
                x = tf.layers.average_pooling2d(x, (1, 3), (1, 3), padding="VALID", name="Conv/AvgPool")
                print(x.shape)
                x = U.flattenallbut0(x)
                print(x.shape)
                x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            else:
                raise NotImplementedError

            tf.summary.histogram("cnn/lin/output", x)

        # batch normalization
        x = tf.layers.batch_normalization(x, training=self.is_training)
        features_relgoal = tf.layers.batch_normalization(features_relgoal, training=self.is_training)
        features_vel     = tf.layers.batch_normalization(features_vel,     training=self.is_training)
        if self.DIRECT_AGENT_OBS:
            features_relobst = tf.layers.batch_normalization(features_relobst, training=self.is_training)
        self.batch_norm_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        all_features = [x, features_relgoal, features_vel]
        if self.DIRECT_AGENT_OBS:
            all_features.append(features_relobst)
        x = tf.concat(all_features, axis=-1)

        x = tf.check_numerics(x, message="after concat")
# 

#         x = tf.nn.relu(tf.layers.dense(x, 256, name='merged_lin', kernel_initializer=U.normc_initializer(1.0)))

        def build_critic_net(inputs, scope):
            with tf.variable_scope(scope):
                dl1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=128,
                                                        activation_fn=tf.nn.relu,
                                                        scope='dl1')

                tf.summary.histogram("{}/dl1/output".format(scope), dl1)

                value = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=1,
                                                          activation_fn=None,
                                                          scope='value')  #[:, 0]  # initializer std 1.0

                tf.summary.histogram("{}/value/output".format(scope), value)
                tf.summary.scalar("{}/value/output_max".format(scope), tf.reduce_max(value))
                tf.summary.scalar("{}/value/output_min".format(scope), tf.reduce_min(value))
                tf.summary.scalar("{}/value/target_max".format(scope), tf.reduce_max(self.target_returns))
                tf.summary.scalar("{}/value/target_min".format(scope), tf.reduce_min(self.target_returns))

                return value
        self.value  = build_critic_net(x, 'value_net')

        def build_actor_net(inputs, scope, trainable, CONTINUOUS):
            with tf.variable_scope(scope):
                # Hidden layer
                dl1 = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs=256,
                                                        activation_fn=tf.nn.relu,
                                                        trainable = trainable,
                                                        scope='dl1')
                # Output layer and distribution
                if not CONTINUOUS:
                    action_logits = tf.contrib.layers.fully_connected(inputs=dl1,
                                                            num_outputs=self.num_action * self.num_action_values,
                                                            activation_fn=tf.nn.relu,
                                                            trainable = trainable,
                                                            scope='action_logits')
                    action_logits = tf.reshape(action_logits, (-1, self.num_action, self.num_action_values))
                    # Multinomial distribution (draw one out of num_action_values classes)
                    # if 3 probs [0.4, 0.1, 0.5] and total_count = 1
                    # sample(1) -> [1, 0, 0], or [0, 1, 0], or [0, 0, 1]
                    # prob([1, 0, 0]) -> 0.4
                    # total_count is the amount of draws per iteration. in this case 1 (single action)
                    action_dist = tf.distributions.Categorical(logits=action_logits)
                else:
                    mu = tf.contrib.layers.fully_connected(inputs=dl1,
                                                           num_outputs=self.num_action,
                                                           activation_fn=tf.nn.tanh,
                                                            scope='mu')
                    # adding epsilon here to prevent inf in normal distribution when sigma -> 0
                    sigma = self.EPSILON + tf.contrib.layers.fully_connected(inputs=dl1,
                                                                       num_outputs=self.num_action,
                                                                       activation_fn=tf.nn.softplus,
                                                                       trainable=trainable,
                                                                       scope='sigma')
                    action_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
                param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
                # tensorboard
                tf.summary.histogram("{}/dl1/output".format(scope), dl1)
                if not CONTINUOUS:
                    action_outputs = tf.split(action_logits, self.num_action, axis=1)
                    for action_name, out in zip(self.action_names, action_outputs):
                        tf.summary.histogram("{}/action_logits/output_{}".format(scope, action_name), out)
                else:
                    mu_outputs = tf.split(mu, self.num_action, axis=-1)
                    for action_name, out in zip(self.action_names, mu_outputs):
                        tf.summary.histogram("{}/mu/output_{}".format(scope, action_name), out)
                    sigma_outputs = tf.split(sigma, self.num_action, axis=-1)
                    for action_name, out in zip(self.action_names, sigma_outputs):
                        tf.summary.histogram("{}/sigma/output_{}".format(scope, action_name), out)
                # ---
                return action_dist, param 

        pi, pi_param = build_actor_net(x, 'actor_net', trainable= True, CONTINUOUS=args.continuous)
        old_pi, old_pi_param = build_actor_net(x, 'old_actor_net', trainable=False, CONTINUOUS=args.continuous)
        self.syn_old_pi = [oldp.assign(p) for p, oldp in zip(pi_param, old_pi_param)]

        single_sample = tf.squeeze(pi.sample(1), axis=0)
        if DISCRETE:
            self.sample_op = single_sample # one_hot
            self.best_action_op = tf.one_hot(tf.argmax(tf.squeeze(pi.probs, axis=0), axis=-1), self.num_action_values) # one_hot
        else:
            self.sample_op = tf.clip_by_value(single_sample, self.action_bound[0][0], self.action_bound[1][0])
            self.best_action_op = tf.clip_by_value(pi.mean(), self.action_bound[0][0], self.action_bound[1][0])
        # tensorboard
        single_sample_outputs = tf.split(single_sample, self.num_action, axis=1)
        for action_name, out in zip(self.action_names, single_sample_outputs):
            tf.summary.histogram("ActionDistribution/single_sample_{}".format(action_name), out)


        # Losses
        with tf.variable_scope('critic_loss'):
            diff_ypred_y = self.target_returns - self.value
            self.critic_loss_ = tf.square(diff_ypred_y)
            CLIP_VALUE_OPTIM = True
            if CLIP_VALUE_OPTIM:
                valueclipped = self.old_value + tf.clip_by_value(self.value - self.old_value, 
                                                                 -self.cliprange, self.cliprange)
                self.clipped_critic_loss = tf.square(self.target_returns - valueclipped)
                self.critic_loss_ = tf.maximum(self.critic_loss_, self.clipped_critic_loss)
            self.critic_loss = tf.reduce_mean(self.critic_loss_)

            self.critic_loss = tf.check_numerics(self.critic_loss, message="after critic_loss")

        with tf.variable_scope('actor_loss'):
            self.entropy = pi.entropy()
            batch_entropy = tf.reduce_mean(self.entropy)
            ratio = pi.prob(self.a) / ( old_pi.prob(self.a) + self.EPSILON )  #(old_pi.prob(self.a)+ 1e-5)
#             ratio = tf.exp( pi.log_prob(self.a) -  old_pi.log_prob(self.a) ) # new / old  #(old_pi.prob(self.a)+ 1e-5)
            pg_losses= -self.advantage * ratio
            pg_losses2 = -self.advantage * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.actor_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)) - batch_entropy * self.entropy_coeff
            self.actor_loss = tf.check_numerics(self.actor_loss, message="after actor_loss")

        # diagnostics
#         if args.continuous:
            # entropy is not implemented for multinomial distribution
        if True:
            self.kl = tf.distributions.kl_divergence(pi, old_pi)
            tf.summary.histogram("Diagnostics/KL", self.kl)
            tf.summary.scalar("Diagnostics/MinibatchAvgKL", tf.reduce_mean(self.kl))
            tf.summary.histogram("Diagnostics/Entropy", self.entropy)
            tf.summary.scalar("Diagnostics/MinibatchAvgEntropy", batch_entropy)
        #explained variance 1 = perfect, 0-1 good, 0 = might as well have predicted 0, < 0 worse than predicting 0
        def reduce_variance(x):
            """ reduce all but batch dim,
             input shape (batch_size, N)
             result shape (batch_size, ) """
            means = tf.reduce_mean(x, keepdims=True)
            sqdev = tf.square(x - means)
            return tf.reduce_mean(sqdev)
        self.ev = 1 - reduce_variance(diff_ypred_y) / reduce_variance(self.target_returns)
        tf.summary.scalar("Diagnostics/MinibatchExplainedVariance", self.ev)

        # add tensorboard summaries
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            variable_summaries(var)
        self.merged_summaries = tf.summary.merge_all()

    def save_model(self, sess, saver, time_step):
        checkpoint_path = self.checkpoint_path
        print('............save model to {} ............'.format(checkpoint_path))
        saver.save(sess, checkpoint_path + '/'+ 'model', global_step=time_step )

    def save_best_model(self, sess, saver, time_step):
        checkpoint_path = self.checkpoint_path + '/best_model'
        print('............save model to {}............'.format(checkpoint_path))
        saver.save(sess, checkpoint_path + '/'+ 'model', global_step=time_step )

    def load_model(self, sess, saver):
        checkpoint_path = self.checkpoint_path + '/'
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('.............Model restored to global from {}.............'.format(checkpoint_path))
        else:
            raise IOError('................No model is found in {}.................'.format(checkpoint_path))

    def load_best_model(self, sess, saver):
        checkpoint_path = self.checkpoint_path + '/best_model/'
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print('.............Model restored to global from {}.............'.format(checkpoint_path))
        else:
            raise IOError('................No model is found in {}.................'.format(checkpoint_path))

    def choose_action(self, s, sess):
        if self.DIRECT_AGENT_OBS:
            action, value = sess.run([self.sample_op, self.value], {self.s: s[1], self.s_conv: s[0], self.s_relobst: s[2], self.is_training: False})
        else:
            action, value = sess.run([self.sample_op, self.value], {self.s: s[1], self.s_conv: s[0], self.is_training: False})
        return action, value

    def choose_action_deterministic(self, s, sess):
        if self.DIRECT_AGENT_OBS:
            action, value = sess.run([self.best_action_op, self.value], {self.s: s[1], self.s_conv: s[0], self.s_relobst: s[2], self.is_training: False})
        else:
            action, value = sess.run([self.best_action_op, self.value], {self.s: s[1], self.s_conv: s[0], self.is_training: False})
        return action, value

    def get_value(self, s, sess):
        if self.DIRECT_AGENT_OBS:
            return sess.run(self.value, {self.s: s[1], self.s_conv: s[0], self.s_relobst: s[2], self.is_training: False})[:,0]
        else:
            return sess.run(self.value, {self.s: s[1], self.s_conv: s[0], self.is_training: False})[:,0]





