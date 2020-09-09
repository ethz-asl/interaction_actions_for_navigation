import tensorflow as tf
import numpy as np
import os
import  baselines.common.tf_util as U

# for versioning. Increment if changes make the model no longer backwards-compatible.
VAE_model_type_letter__ = "A"

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


class VAE(object):
    def __init__(self, input_shape, scope, args):
        assert len(input_shape) == 3
        self.input_shape = input_shape # (W, H, Channels)
        self.scope = scope
        self.MASKS = args.masks
        self.Z_SIZE = args.z_size
        self.EPSILON = 1e-8
        self.checkpoint_path = args.checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.trained_epochs = tf.Variable(0, dtype=tf.int32, name='trained_epochs', trainable=False)
        self.inc_trained_epochs = self.trained_epochs.assign_add(1)


        ## Build net
        with tf.variable_scope('input'):
            self.x_in = tf.placeholder(name="x_in", dtype="float", shape=(None, ) + self.input_shape) # Batch, W, H, Channels
            self.z_in = tf.placeholder(name="z_in", dtype="float", shape=(None, ) + (self.Z_SIZE,) ) # Batch, Z
            self.mask = tf.placeholder(name="mask", dtype="float", shape=(None, ) + self.input_shape[:-1] + (1,) ) # Batch, W, H, 1
        with tf.variable_scope('is_training'):
            self.is_training = tf.placeholder(tf.bool, name="is_training")
        with tf.variable_scope('kl_tolerance'):
            self.kl_tolerance = tf.placeholder(name="kl_tolerance", dtype=tf.float32)




#     def build_VAE(x_in, mask, is_training, kl_tolerance, Z_SIZE):
        """ 
            x_in (tf.placeholder): input (and target output) of the autoencoder network
            mask (tf.placeholder): is_person mask. Where this mask is True normal reconstruction_loss is computed.
                                where it is False, loss is set to 0.
            is_training (tf.placeholder): is training
            kl_tolerance (scalar, or tf.placeholder): 
            Z_SIZE (scalar): size of the latent z dimension
        """
        is_training = self.is_training
        x = self.x_in
        _7 = 7 if input_shape[0] > 64 else 1 # either 1 or 7 (whether input is lidar or image)
        _3 = 3 if input_shape[0] > 64 else 1 # either 1 or 3 
        _3_else_2 = 3 if input_shape[0] > 64 else 2
        with tf.variable_scope('encoder'):
            print("A0: {}".format(x.shape))
            x = tf.layers.batch_normalization(
                    tf.nn.relu(U.conv2d(x, 64, "l1", [_7, 7], [_3, 3], pad="SAME", summary_tag="Conv/Layer1")), training=is_training)
            print("A1: {}".format(x.shape))
            x = tf.layers.max_pooling2d(x, (_3, 3), (_3, 3), padding="SAME", name="Conv/MaxPool")
            xres = x
            print("A2: {}".format(x.shape))
            x = tf.layers.batch_normalization(
                    tf.nn.relu(U.conv2d(x, 64, "l2", [_3, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer2")), training=is_training)
            print("A3: {}".format(x.shape))
            x = tf.layers.batch_normalization(
                    U.conv2d(x, 64, "l3", [_3, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer3"), training=is_training)
            print("A4: {}".format(x.shape))
            xres2 = x
            x = tf.nn.relu(x + xres)
            x = tf.layers.batch_normalization(
                    tf.nn.relu(U.conv2d(x, 64, "l4", [_3_else_2, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer4")), training=is_training)
            print("A5: {}".format(x.shape))
            x = tf.layers.batch_normalization(
                    U.conv2d(x, 64, "l5", [_3_else_2, 3], [1, 1], pad="SAME", summary_tag="Conv/Layer5"), training=is_training)
            print("A6: {}".format(x.shape))
            x = tf.nn.relu(x + xres2)
            x = tf.layers.average_pooling2d(x, (_3, 3), (_3, 3), padding="SAME", name="Conv/AvgPool")
            endconv_shape = x.shape
            print("A7: {}".format(x.shape))
            x = U.flattenallbut0(x)
            endconv_flat_shape = x.shape
            print("A8: {}".format(x.shape))
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
            print("A9: {}".format(x.shape))

            tf.summary.histogram("encoder/lin/output", x)

        with tf.variable_scope('latent_space'):
            z_mu = tf.nn.sigmoid(tf.layers.dense(x, self.Z_SIZE, name='z_mu', kernel_initializer=U.normc_initializer(1.0)))
            z_logvar = tf.nn.relu(tf.layers.dense(x, self.Z_SIZE, name='z_logvar', kernel_initializer=U.normc_initializer(1.0)))
            z_sigma = tf.exp(z_logvar/2.0)
            z = tf.contrib.distributions.Normal(loc=z_mu, scale=z_sigma)
            x = z.sample(1)[0]
            print("Z: {}".format(x.shape))
            self.z_mu = z_mu
            self.z_sigma = z_sigma
            self.z = z
            self.z_sample = x

        def build_decoder(z, is_training=self.is_training, output_shape=self.input_shape, scopename="decoder", reuse=False):
            with tf.variable_scope(scopename, reuse=reuse) as scope:
                x = z
                x = tf.nn.relu(tf.layers.dense(x, 512, name='z_inv', kernel_initializer=U.normc_initializer(1.0)))
                print("A9: {}".format(x.shape))
                x = tf.nn.relu(tf.layers.dense(x, endconv_flat_shape[1], name='lin_inv', kernel_initializer=U.normc_initializer(1.0)))
                print("A8: {}".format(x.shape))
                x = tf.reshape(x, (-1, endconv_shape[1], endconv_shape[2], endconv_shape[3]))
                print("A7: {}".format(x.shape))
                # 'opposite' of average_pooling2d with stride
        #         x = tf.image.resize_nearest_neighbor(x, (1*x.shape[1], 3*x.shape[2]), align_corners=True)
                x = tf.layers.conv2d_transpose(x, 64, (_3, 3), (_3, 3), activation=tf.nn.relu, padding="SAME", name="avgpool_inv")
                xres2 = x
                print("A6: {}".format(x.shape))
                x = tf.layers.batch_normalization(
                        tf.layers.conv2d_transpose(x, 64, (_3_else_2, 3), (1, 1), activation=tf.nn.relu, padding="SAME", name="l5_inv"), training=is_training)
                print("A5: {}".format(x.shape))
                x = tf.layers.batch_normalization(
                        tf.layers.conv2d_transpose(x, 64, (_3_else_2, 3), (1, 1), activation=tf.nn.relu, padding="SAME", name="l4_inv"), training=is_training)
                x = tf.nn.relu(x + xres2)
                xres = x
                print("A4: {}".format(x.shape))
                x = tf.layers.batch_normalization(
                        tf.layers.conv2d_transpose(x, 64, (_3, 3), (1, 1), activation=tf.nn.relu, padding="SAME", name="l3_inv"), training=is_training)
                print("A3: {}".format(x.shape))
                x = tf.layers.batch_normalization(
                        tf.layers.conv2d_transpose(x, 64, (_3, 3), (1, 1), activation=tf.nn.relu, padding="SAME", name="l2_inv"), training=is_training)
                print("A2: {}".format(x.shape))
                x = tf.nn.relu(x + xres)
                x = tf.layers.conv2d_transpose(x, 64, (_3, 3), (_3, 3), activation=tf.nn.relu, padding="SAME", name="maxpool_inv")
                print("A1: {}".format(x.shape))
                x = tf.layers.batch_normalization(
                        tf.layers.conv2d_transpose(x, output_shape[2], (_7, 7), (_3, 3), activation=tf.nn.relu, padding="SAME", name="l1_inv"), training=is_training)
                print("A0: {}".format(x.shape))
                y = x
            return y

        self.y = build_decoder(self.z_sample)
        # This must be done before creating the pure decoder, or tf will expect z_in to be fed
        self.batch_norm_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # Create a separate decoder network with same variables fed by placeholder, not encoder
        # for off-training reconstruction
        self.reconstruction = build_decoder(self.z_in, reuse=True)






        # Losses
        with tf.variable_scope('reconstruction_loss'):
            self.avg_rec_abs_error = tf.reduce_mean(tf.abs(self.x_in - self.y),
                    reduction_indices=[0,1,2]) # per channel
#             reconstruction_s_e = tf.square((self.x_in - self.y) / 255) # reconstruction square of normalized error
            reconstruction_s_e = tf.log(tf.cosh((self.x_in - self.y) / 255)) # reconstruction square of normalized error
            if self.MASKS: # apply mask (W, H) to p.pixel error (Batch, W, H, Channels)
                reconstruction_s_e  = tf.boolean_mask(reconstruction_s_e, self.mask) 
            reconstruction_loss = tf.reduce_mean(reconstruction_s_e, reduction_indices=[1,2,3]) # per example
            self.reconstruction_loss = tf.reduce_mean(reconstruction_loss) # average over batch

            # kl loss (reduce along z dimensions)
            kl_loss = - 0.5 * tf.reduce_mean(
                    (1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar)), 
                    reduction_indices = 1
                    ) 
            kl_loss = tf.maximum(kl_loss, self.kl_tolerance) # kl_loss per example
            self.kl_loss = tf.reduce_mean(kl_loss) # batch kl_loss

            self.loss = self.reconstruction_loss + self.kl_loss

        # add tensorboard summaries
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            variable_summaries(var)
        self.merged_summaries = tf.summary.merge_all()

        # A placeholder for adding arbitrary images to tensorboard
        self.image_tensor = tf.placeholder(name="image", dtype="float", shape=(None, 1000, 1000, 4)) # Batch, W, H, Channels
        self.image_summary = tf.summary.image("Reconstructions/val", self.image_tensor)
        self.image_tensor2 = tf.placeholder(name="image2", dtype="float", shape=(None, 1000, 1000, 4)) # Batch, W, H, Channels
        self.image_summary2 = tf.summary.image("Reconstructions/valtarget", self.image_tensor2)



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


    def encode(self, x, sess):
        return sess.run(self.z_sample, {self.x_in: x, self.is_training: False})

    def decode(self, z, sess):
        return sess.run(self.reconstruction, {self.z_in: x, self.is_training: False})

    def reconstruct(self, x, sess):
        return sess.run(self.y, {self.x_in: x, self.is_training: False})
