# RL environment for Pepper Navigation

![RL environment for Pepper Robot](https://github.com/ethz-asl/asl_pepper/raw/master/wiki/pepper_rl.png "Pepper RL Environment")
## Set up

Follow the install procedures on this repository's [README file.](https://github.com/ethz-asl/asl_pepper/blob/master/README.md)

## Tutorials

### Training a Model

```
source ~/pepper_ws/devel/setup.bash
source ~/peppervenv/bin/activate
roscd asl_pepper_motion_planning/scripts
python trainPPO.py --mode BOUNCE --map-name office_full --max-episode-length 128
```

Model checkpoints and summaries will be saved in ```~/PPO```

Note: ```python trainPPO.py --help``` can be used to outline possible arguments for the training script.

### Visualizing Training Progress with Tensorboard

```
source ~/peppervenv/bin/activate
tensorboard --logdir ~/PPO/summaries
# (open tensorboard uri in browser)
```

### Testing Model Output

To find the names of possible models to test, type
```
ls ~/PPO/models
```

pick a model name, for example ```PepperRLSim_MlpPolicy_Ed_run_20190704_161928_835283```

```
source ~/pepper_ws/devel/setup.bash
source ~/peppervenv/bin/activate
roslaunch asl_pepper_motion_planning test_rl.launch mapname:=office_full script_args:="--mode BOUNCE --resume-from ~/PPO/models/PepperRLSim_MlpPolicy_Ed_run_20190704_161928_835283"
```

in a separate terminal:

```
source ~/pepper_ws/devel/setup.bash
rviz
```

To visualize the simulated robots, in the rviz 'Displays' tab:
- under Global Options > Fixed Frame select ```odom```
- Press the 'Add' button, select topics: ```/map```, ```/agent_cmd_vels```, ```/agent_goals``` and any other topics you may want to visualize.


## Code

The model and training scripts are mainly found in ```asl_pepper_motion_planning/scripts```
```
roscd asl_pepper_motion_planning/scripts
```

in ```scripts/```:
- ```trainPPO.py``` contains the code for the training regime.
- ```PPO.py``` contains most of the code defining the model architecture.

The Pepper simulator used in training is found in ```asl_pepper_2d_simulator/python```
```
roscd asl_pepper_2d_simulator/python
```

in ```python/```:
- ```pepper_2d_simulator.py``` contains most of the code defining the behavior of the environment, in particular the ```step()``` method.
