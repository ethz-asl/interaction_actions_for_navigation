import argparse
import os

try:
    import rospy
    ROS_FOUND = True
except ImportError:
    ROS_FOUND = False


def parse_sim_args():
    # Arguments
    parser = argparse.ArgumentParser(description='Test node for the pepper RL path planning algorithm.')
    populate_sim_args(parser)
    # add external args
#     from xxx import parse_xxx_args
#     ARGS, unknown = parse_xxx_args(ignore_unknown=True, parser=parser)
    ARGS, unknown_args = parser.parse_known_args()

    # deal with unknown arguments
    # ROS appends some weird args, ignore those, but not the rest
    if unknown_args:
        if ROS_FOUND:
            non_ros_unknown_args = rospy.myargv(unknown_args)
        else:
            non_ros_unknown_args = unknown_args
        if non_ros_unknown_args:
            print("unknown arguments:")
            print(non_ros_unknown_args)
            parser.parse_args(args=["--help"])
            raise ValueError

    return ARGS


def populate_sim_args(parser):
    argscriptdir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--map-folder', type=str, default=os.path.join(argscriptdir, "../maps"),)
    parser.add_argument('--map-name', type=str, default="asl",)
    parser.add_argument('--map-downsampling-passes', type=int, default=0,)
    # planning args
    parser.add_argument('--n-trials', type=int, default=1000,)
    parser.add_argument('--max-depth', type=int, default=10,)
    single_action = parser.add_mutually_exclusive_group()
    single_action.add_argument('--only-intend', action='store_true',)
    single_action.add_argument('--only-say', action='store_true',)
    single_action.add_argument('--only-nudge', action='store_true',)
    # scenario args
    parser.add_argument(
        '--scenario-folder',
        type=str,
        default=os.path.join(argscriptdir, "../scenarios"),
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default="aslquiet",
        help="""Which named scenario to load from the scenarios/ folder.
        The scenario defines initial agents and their state""",
    )
    # utilities args
    parser.add_argument('--debug', action='store_true',)
    parser.add_argument('--plotless', action='store_true',)
