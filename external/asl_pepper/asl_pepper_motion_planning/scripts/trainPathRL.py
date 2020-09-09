from trainPPO import *
from smallMlp import smallMlp

from pepper_2d_iarlenv import IARLEnv, populate_iaenv_args, check_iaenv_args
from pepper_2d_iarlenv import MultiIARLEnv, parse_training_args
if __name__ == '__main__':
    tf.reset_default_graph()

    env_populate_args_func = populate_iaenv_args
    env_check_args_func = check_iaenv_args
    env_type = MultiIARLEnv

    # args
    args, _ = parse_training_args(
      ignore_unknown=False,
      env_populate_args_func=env_populate_args_func,
      env_name="IARLEnv"
      )
    if not args.force_publish_ros:
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

