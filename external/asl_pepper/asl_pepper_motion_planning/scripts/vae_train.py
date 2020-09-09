from __future__ import print_function
from socket import gethostname
import io
import numpy as np
import os
import random
import skimage
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from timeit import default_timer as timer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from VAE import *

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return [skimage.io.imread(buf)]

def parse_VAE_args(ignore_unknown=True, parser=None):
    import argparse
    if parser is None:
        parser = argparse.ArgumentParser()
    # Training args
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument( '--resume-latest', action='store_true',
            help='loads latest compatible saved model and continues training from there.',)
    resume.add_argument('--resume-from', type=str, default='',
            help='loads a specific checkpoint',)
    parser.add_argument('--checkpoint-root-dir', type=str, default='~/VAE/models', help='root directory for storing models of all runs')
    parser.add_argument('--summary-root-dir', type=str, default='~/VAE/summaries', help='root directory for storing logs of all runs')
    parser.add_argument('--dataset-dir', type=str, default='~/VAEDatasets/NYC1024', help='directory containing train_frames/ and val_frames/')
    parser.add_argument('--interactive', '-i', action="store_true", help="Create graph, loads model, and hands over control to user")
    trparser = parser.add_argument_group('Training', 'parameters for the model training.')
    trparser.add_argument('--model', type=str, default='VAE')
    trparser.add_argument('--masks', action="store_true", help='Apply masks to reconstruction loss, making selected areas significant')
    trparser.add_argument('--learning-rate', '--lr', type=float, default=0.01)
    trparser.add_argument('--z-size', type=int, default=16)
    trparser.add_argument('--kl-tolerance', '--ent', type=float, default=0.01)
    trparser.add_argument('--minibatch-size', type=int, default=32)
    trparser.add_argument('--random-seed', type=int, default=np.random.randint(1000000))
    trparser.add_argument('--clip-gradients', type=float, default=0.0, help='Max gradient norm, set to 0 to disable')
    trparser.add_argument( '--progress-to-file', action='store_true', help='also log progress in a file. useful for cluster execution.',)

    if ignore_unknown:
        args, unknown_args = parser.parse_known_args()
    else:
        args = parser.parse_args()
        unknown_args = []

    # Paths --------------------------------------------
    # expanduser in directory paths
    args.checkpoint_root_dir = os.path.expanduser(args.checkpoint_root_dir)
    args.summary_root_dir = os.path.expanduser(args.summary_root_dir)
    args.dataset_dir = os.path.expanduser(args.dataset_dir)
    args.train_im_dir = os.path.join(args.dataset_dir, 'train')
    args.val_im_dir = os.path.join(args.dataset_dir, 'val')

    # Create full paths for this run
    from datetime import datetime
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_policy_folder =  args.model + '_' + VAE_model_type_letter__ 
    args.run_folder = 'run_' + date_time  + '_' + "{}".format(args.random_seed)
    args.summary_path = args.summary_root_dir + '/' + env_policy_folder + '_' + args.run_folder
    args.checkpoint_path = args.checkpoint_root_dir + '/' + env_policy_folder + '_' + args.run_folder

    # if resuming, change checkpoint path to previous one.
    # however, keep logging path the same
    # resume from latest
    if args.resume_latest:
        # find compatible models
        from os import listdir
        models = listdir(args.checkpoint_root_dir)
        compatible_models = sorted([model_dir for model_dir in models if env_policy_folder in model_dir])
        args.latest_model_folder = None
        if compatible_models:
            args.latest_model_folder = compatible_models[-1]
        # change to latest model
        if args.latest_model_folder is None:
            print(env_policy_folder)
            raise ValueError("No model to resume from. Found models: {} of which compatible models: {}".format(
                models, compatible_models))
        args.resume_from = args.checkpoint_root_dir + '/' + args.latest_model_folder
    if args.resume_from != '':
        print("Resuming model {}".format(args.resume_from))
        if env_policy_folder not in args.resume_from:
            print("WARNING: detected potential incompatibility between model to load and current model")
            print(env_policy_folder)
            print(args.resume_from)
        args.checkpoint_path = args.resume_from
    # ---------------------------------------------------

    return args, unknown_args

def chunks(l, n):
    """ Create a function called "chunks" with two arguments, l and n:
        which splits a list l into chunks of size n (for last chunk, might be less) """
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def len_chunks(l, n):
    return len(range(0, len(l), n))



# MAIN -----------------------------------------
if __name__ == '__main__':
    tf.reset_default_graph()

    args, _ = parse_VAE_args(ignore_unknown=False)
    print(args)

    summary_writer = tf.summary.FileWriter(args.summary_path)

    tf.set_random_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # DATASETS
    input_shape = (459, 837, 3)
    # keras datagen
    train_datagen = ImageDataGenerator(
            rescale=None,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=None)
    train_generator = train_datagen.flow_from_directory(
        directory=args.train_im_dir,
        target_size=input_shape[:2],
        color_mode="rgb",
        batch_size=args.minibatch_size,
        class_mode=None,
        shuffle=True,
        seed=args.random_seed,
    )
    val_generator = val_datagen.flow_from_directory(
        directory=args.val_im_dir,
        target_size=input_shape[:2],
        color_mode="rgb",
        batch_size=args.minibatch_size,
        class_mode=None,
        shuffle=True,
        seed=args.random_seed,
    )
    file_names = train_generator.filenames
    val_file_names = val_generator.filenames


    # VARIABLES
    MAX_EPOCHS = 100000
    images_per_epoch = len(file_names)

    # MODEL
    scope = "train_vae"
    # input_shape = (4, 1080, 1)
    vae = VAE(input_shape, scope, args)
    # OPTIMIZER
    adam = tf.train.AdamOptimizer(args.learning_rate, epsilon=1e-5)
    def apply_gradients_clipped(adamoptimizer, loss, max_grad_norm=args.clip_gradients):
        grads_and_var = adamoptimizer.compute_gradients(loss)
        grads, var = zip(*grads_and_var)
        if not args.clip_gradients == 0:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))
        return adamoptimizer.apply_gradients(grads_and_var)
    train_op = apply_gradients_clipped(adam, vae.loss)

    print("----------- CREATED GRAPH")

    # Run training
    with tf.Session() as sess:
        if args.interactive:
            sess = tf.Session() # this one won't get closed when we leave the script and hand over
        saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        best_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
        sess.run(tf.global_variables_initializer())
        if args.resume_from != '':
            vae.load_model(sess, saver)

        """ Iterate over batches,
        [batch_size] steps (potentially several episodes) are collected into a batch
        it is then split into minibatches, each used for the optimization
        """

        # log initial state of variables
        summary_writer.add_graph(sess.graph)
        trained_epochs = sess.run(vae.trained_epochs)
        best_val_loss = None
        summary = tf.Summary()
        summary.value.add(tag="Machine/{}".format(os.path.expandvars("$MACHINE_NAME")), simple_value=float(args.random_seed))
        summary.value.add(tag="HyperParams/RandomSeed", simple_value=float(args.random_seed))
        summary.value.add(tag="HyperParams/Masks", simple_value=float(args.masks))
        summary.value.add(tag="HyperParams/LearningRate", simple_value=float(args.learning_rate))
        summary.value.add(tag="HyperParams/KLTolerance", simple_value=float(args.kl_tolerance))
        summary.value.add(tag="HyperParams/MaxGradNorm", simple_value=float(args.clip_gradients))
        summary.value.add(tag="HyperParams/MinibatchSize", simple_value=float(args.minibatch_size))
        summary_writer.add_summary(summary, trained_epochs)
        summary_writer.flush()
        feed_dict = {}
        feed_dict[vae.x_in] = np.zeros((1,) + input_shape)
        feed_dict[vae.is_training] = False
        merged_summaries = sess.run(vae.merged_summaries, feed_dict=feed_dict)
        summary_writer.add_summary(merged_summaries, trained_epochs)
        summary_writer.flush()

        def generate_reconstructions():
            GRID_SIDE = 3
            # todo func args instead of val_file_names, args.val_im_dir
            recstr_file_names = val_file_names[:GRID_SIDE**2]
            images = np.array([skimage.io.imread(os.path.join(args.val_im_dir, filename))[:input_shape[0], :input_shape[1], :] for filename in recstr_file_names])
            reconstructions = vae.reconstruct(images, sess)
            reconstructions = np.clip(reconstructions, 0, 255).astype(int)
            # Create a figure to contain the plot.
            figure = plt.figure(figsize=(10,10))
            for i in range(GRID_SIDE**2):
                # Start next subplot.
                plt.subplot(GRID_SIDE, GRID_SIDE, i + 1, title=recstr_file_names[i])
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(reconstructions[i])
            plt.tight_layout()
            return plot_to_image(figure)

        def generate_reconstruction_targets():
            GRID_SIDE = 3
            # todo func args instead of val_file_names, args.val_im_dir
            recstr_file_names = val_file_names[:GRID_SIDE**2]
            images = np.array([skimage.io.imread(os.path.join(args.val_im_dir, filename))[:input_shape[0], :input_shape[1], :] for filename in recstr_file_names])
            # Create a figure to contain the plot.
            figure = plt.figure(figsize=(10,10))
            for i in range(GRID_SIDE**2):
                # Start next subplot.
                plt.subplot(GRID_SIDE, GRID_SIDE, i + 1, title=recstr_file_names[i])
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(images[i])
            plt.tight_layout()
            return plot_to_image(figure)



        if args.progress_to_file:
            print("Host: {}".format(gethostname())
             , file=open(os.path.expanduser("~/{}.txt".format(args.run_folder)), "a+") )

        for epoch in range(MAX_EPOCHS):
            epoch_batch_gen_time = 0
            epoch_optim_time = 0
            epoch_training_loss = 0
            epoch_training_reconstruction_loss = 0
            epoch_training_kl_loss = 0
            batches_per_epoch = len(train_generator)
            for i in range(batches_per_epoch):
                if not os.path.exists(args.checkpoint_path):
                    raise IOError("Model folder {} not found".format(args.checkpoint_path))
                trained_epochs = sess.run(vae.trained_epochs)

                # Generate batch
                tic = timer()
                image_batch = train_generator.next()
                toc = timer()
                epoch_batch_gen_time += toc-tic

                # Process batch
                tic = timer()

                feed_dict = {}
                feed_dict[vae.x_in] = image_batch
                if args.masks:
                    feed_dict[vae.masks] = np.ones((image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1))
                feed_dict[vae.is_training] = True
                feed_dict[vae.kl_tolerance] = args.kl_tolerance

#                 print("Processing 1 batch")
                training_loss, training_reconstruction_loss, training_kl_loss, _, _  = sess.run(
                        [vae.loss, vae.reconstruction_loss, vae.kl_loss, train_op, vae.batch_norm_update_op], 
                        feed_dict=feed_dict)
                epoch_training_loss += training_loss
                epoch_training_reconstruction_loss += training_reconstruction_loss
                epoch_training_kl_loss += training_kl_loss

                toc = timer()
                epoch_optim_time += toc-tic
#                 print("batch {} / {} - gen : {:.2f}, optim : {:.2f}".format(
#                     i, len(file_names)//args.minibatch_size,epoch_batch_gen_time, epoch_optim_time))
            # Summed batch training loss to average epoch loss
            epoch_training_loss = epoch_training_loss / batches_per_epoch
            epoch_training_reconstruction_loss = epoch_training_reconstruction_loss / batches_per_epoch
            epoch_training_kl_loss = epoch_training_kl_loss / batches_per_epoch
            print("Optim complete - training loss: {}, batch gen time: {:.2f} s, optim time: {:.2f} s".format(
                epoch_training_loss, epoch_batch_gen_time, epoch_optim_time))
            if args.progress_to_file:
                print("Optim complete - training loss: {}, batch gen time: {:.2f} s, optim time: {:.2f} s".format(
                    epoch_training_loss, epoch_batch_gen_time, epoch_optim_time)
                 , file=open(os.path.expanduser("~/{}.txt".format(args.run_folder)), "a+") )

            # end of epoch logging
            trained_epochs = sess.run(vae.inc_trained_epochs)


            # Validation
            valepoch_batch_gen_time = 0
            valepoch_optim_time = 0
            epoch_val_loss = 0
            epoch_val_reconstruction_loss = 0
            epoch_val_kl_loss = 0
            epoch_avg_rec_error = np.zeros((input_shape[2]))
            batches_per_val = len(val_generator)
            for i in range(batches_per_val):

                # Generate batch
                tic = timer()
                image_batch = val_generator.next()
                toc = timer()
                valepoch_batch_gen_time += toc-tic

                # Process batch
                tic = timer()

                feed_dict = {}
                feed_dict[vae.x_in] = image_batch
                if args.masks:
                    feed_dict[vae.masks] = np.ones((image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 1))
                feed_dict[vae.is_training] = False
                feed_dict[vae.kl_tolerance] = args.kl_tolerance

#                 print("Processing 1 batch")
                val_loss, val_reconstruction_loss, val_kl_loss, avg_rec_error = sess.run(
                        [vae.loss, vae.reconstruction_loss, vae.kl_loss, vae.avg_rec_abs_error],
                        feed_dict=feed_dict)
                epoch_val_loss += val_loss
                epoch_val_reconstruction_loss += val_reconstruction_loss
                epoch_val_kl_loss += val_kl_loss
                epoch_avg_rec_error += avg_rec_error

                toc = timer()
                valepoch_optim_time += toc-tic
#                 print("batch {} / {} - gen : {:.2f}, optim : {:.2f}".format(
#                     i, len(file_names)//args.minibatch_size,valepoch_batch_gen_time, valepoch_optim_time))
            # Summed batch training loss to average val loss
            epoch_val_loss = epoch_val_loss / batches_per_val
            epoch_val_reconstruction_loss = epoch_val_reconstruction_loss / batches_per_val
            epoch_val_kl_loss = epoch_val_kl_loss / batches_per_val
            print("Val complete - training loss: {}, batch gen time: {:.2f} s, optim time: {:.2f} s".format(
                epoch_val_loss, valepoch_batch_gen_time, valepoch_optim_time))
            if args.progress_to_file:
                print("Val complete - training loss: {}, batch gen time: {:.2f} s, optim time: {:.2f} s".format(
                    epoch_val_loss, valepoch_batch_gen_time, valepoch_optim_time)
                 , file=open(os.path.expanduser("~/{}.txt".format(args.run_folder)), "a+") )

            # log
            summary = tf.Summary()
            summary.value.add(tag='EpochLosses/TrainLoss', simple_value=float(epoch_training_loss))
            summary.value.add(tag='EpochLosses/TrainReconstructionLoss', simple_value=float(epoch_training_reconstruction_loss))
            for i, ch_err in enumerate(epoch_avg_rec_error):
                summary.value.add(tag='EpochLosses/TrainReconstructionError_{}'.format(i),
                        simple_value=float(ch_err))
            summary.value.add(tag='EpochLosses/TrainKLLoss', simple_value=float(epoch_training_kl_loss))
            summary.value.add(tag='EpochLosses/ValLoss', simple_value=float(epoch_val_loss))
            summary.value.add(tag='EpochLosses/ValReconstructionLoss', simple_value=float(epoch_val_reconstruction_loss))
            summary.value.add(tag='EpochLosses/ValKLLoss', simple_value=float(epoch_val_kl_loss))
            summary_writer.add_summary(summary, trained_epochs)
            summary_writer.flush()

            # Save best model
            if best_val_loss is None:
                best_val_loss = epoch_val_loss
            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
                vae.save_best_model(sess, best_saver, trained_epochs)
                summary = tf.Summary()
                summary.value.add(tag='Batches/Overall_Best_Val_Loss', simple_value=float(best_val_loss))
                summary_writer.add_summary(summary, trained_epochs)
                summary_writer.flush()

            # Also Save model every n_steps
            save_every_n_epochs = 128 # 2**17
            if trained_epochs % save_every_n_epochs == 0 or trained_epochs == 1:
                vae.save_model(sess, saver, trained_epochs)


            # Log internals progressively less often
            intern_summary_every_n_epochs = 1
            if trained_epochs > 100000:
                intern_summary_every_n_epochs = 10
            if trained_epochs % intern_summary_every_n_epochs == 0:
                # use the last feed dict batch from validation because lazy
                merged_summaries = sess.run(vae.merged_summaries, feed_dict=feed_dict)
                summary = tf.Summary()
                summary_writer.add_summary(summary, trained_epochs)
                summary_writer.add_summary(merged_summaries, trained_epochs)
                summary_writer.flush()

            # Log reconstructions every so often
            if trained_epochs == 1 or trained_epochs % 1024 == 0:
                if trained_epochs == 1:
                    # targets as a figure in tensorboard
                    summary2 = sess.run(vae.image_summary2,
                            feed_dict={vae.image_tensor2: generate_reconstruction_targets()})
                    summary_writer.add_summary(summary2, trained_epochs)
                # reconstructions as a figure in tensorboard
                summary = sess.run(vae.image_summary,
                        feed_dict={vae.image_tensor: generate_reconstructions()})
                summary_writer.add_summary(summary, trained_epochs)
                # flush
                summary_writer.flush()


            # log timings
            if trained_epochs == 1 or trained_epochs % 10 == 0:
                summary = tf.Summary()
                summary.value.add(tag='Timings/BatchGenTime', simple_value=float(epoch_batch_gen_time))
                summary.value.add(tag='Timings/OptimTime', simple_value=float(epoch_optim_time))
                summary.value.add(tag='Timings/PerImage_BatchGenTime', simple_value=float(epoch_batch_gen_time/images_per_epoch))
                summary.value.add(tag='Timings/PerImage_OptimTime', simple_value=float(epoch_optim_time/images_per_epoch))
                summary_writer.add_summary(summary, trained_epochs)
                summary_writer.flush()

            if args.interactive:
                break

