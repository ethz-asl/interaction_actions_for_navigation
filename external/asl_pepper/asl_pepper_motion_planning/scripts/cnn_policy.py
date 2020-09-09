import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

class CnnPolicy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        ob_2 = U.get_placeholder(name="ob_2", dtype=tf.float32, shape=[sequence_length] + [5]) # observations to feed in after convolutions
        ob_2_fc = tf.nn.relu(tf.layers.dense(ob_2, 64, name='s2_preproc', kernel_initializer=U.normc_initializer(1.0)))

        x = ob / 25.
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [2, 8], [1, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [2, 4], [1, 2], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            x = tf.concat([x, ob_2_fc], axis=-1)
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 32, "l1", [2, 8], [1, 4], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [2, 4], [1, 2], pad="VALID"))
            x = tf.nn.relu(U.conv2d(x, 64, "l3", [2, 3], [1, 1], pad="VALID"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            x = tf.concat([x, ob_2_fc], axis=-1)
        else:
            raise NotImplementedError

        x = tf.nn.relu(tf.layers.dense(x, 256, name='merged_lin', kernel_initializer=U.normc_initializer(1.0)))
        logits = tf.layers.dense(x, pdtype.param_shape()[0], name='logits', kernel_initializer=U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = tf.layers.dense(x, 1, name='value', kernel_initializer=U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob, ob_2], [ac, self.vpred])

    def act(self, stochastic, ob):
        ob_1 = ob[0] # laser state
        ob_2 = ob[1] # goal position and robot vel state
        ac1, vpred1 =  self._act(stochastic, ob_1, ob_2)
        return ac1, vpred1
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
    def load_variables(self, path):
        saver = tf.train.Saver()
        sess = U.get_session()
        saver.restore(sess, path)
    def save_variables(self, path):
        saver = tf.train.Saver()
        sess = U.get_session()
#         variables = self.get_variables()
        import os
        dirname = os.path.dirname(path)
        if any(dirname):
            os.makedirs(dirname, exist_ok=True)
        saver.save(sess, path)

