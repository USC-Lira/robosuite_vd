"""
A script to collect a batch of human demonstrations.
# NOTE(dhanush) : This is meant for collecting demonstrations using the Meta Quest Controller with gaze
The demonstrations can be played back using the `playback_demonstrations_from_hdf5.py` script.
"""

import argparse
import datetime
import json
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.wrappers.data_collection_wrapper import DataCollectionWrapper_gaze
from robosuite.utils.oculus_controller import VRPolicy
import robosuite.utils.transform_utils as T

from robosuite.gaze_stuff.gaze_socket_client import SimpleClient
from robosuite.gaze_stuff.gaze_data_utils import gaze_data_util
import pdb


def collect_human_trajectory(policy, env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # ---keybaord-stuff------#
        # # Get the newest action
        # action, grasp = input2action(
        #     device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        # )

        # # If action is none, then this a reset so we should break
        # if action is None:
        #     break
        # ---keybaord-stuff------#

        # VR Controller Observation
        obs_dict = {}
        obs_dict["robot_state"] = {}
        # Fixing format
        ee_pos = env.robots[0].controller.ee_pos
        ee_ori_mat = env.robots[0].controller.ee_ori_mat
        ee_euler = T.mat2euler(ee_ori_mat)
        # Feeding the dict
        obs_dict['robot_state']['cartesian_position'] = np.concatenate([ee_pos, ee_euler]) # 6 coords. 3 xyz ee pose, 3 euler angles
        obs_dict['robot_state']['gripper_position'] = 0
        action = policy.forward(obs_dict) # Giving the observation to the VR 

        if policy.__dict__['_state']['buttons']['B']: # Break if a bad demo
            break

        if  not policy.__dict__['_state']['movement_enabled']: # Step only when enabled
            continue

        # Run environment step
        # print(action) #TODO: for checking the clippping done in the VR class
        # pdb.set_trace()


        # Gaze Data from sensor
        gaze_data_dict_adj, gaze_data_raw = gaze_util_obj.gaze_pixels(gaze_client.get_latest_message()) # Pollling Code

        # print(gaze_data_raw)

        # --Format of Data -- #
        # gaze_data_dict_adj['pixel_x']
        # gaze_data_dict_adj['pixel_y']
        # gaze_data_raw['FPOGX']
        # gaze_data_raw['FPOGY']
        #---------------------#

        # Gaze data (FPOGX, FPOGY), in the same format as action
        gaze_data_np = np.array([gaze_data_raw['FPOGX'], gaze_data_raw['FPOGY']], dtype=np.float64)
        
        # env.step(action) # NOTE(dhanush) : THIS WAS MEANT FOR THE ORIGINAL WRAPPER
        env.step(action, gaze_data_np) # NOTE(dhanush) : THIS WAS MEANT FOR THE MODIFIED WRAPPER

        env.render()

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info): #TODO: verify modification made for gaze
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        gazes = []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

            for gi in dic["gaze_infos"]:
                gazes.append(gi['gazes'])

            success = success or dic["successful"]

        if len(states) == 0:
            continue

        # Add only the successful demonstration to dataset
        if success:
            print("Demonstration is successful and has been saved")
            # Delete the last state. This is because when the DataCollector wrapper
            # recorded the states and actions, the states were recorded AFTER playing that action,
            # so we end up with an extra state at the end.
            del states[-1]
            assert len(states) == len(actions)
            assert len(states) == len(gazes)


            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions and gaze
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
            ep_data_grp.create_dataset("gazes", data=np.array(gazes))

        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lifteither")
    # parser.add_argument("--environment", type=str, default="Block_Pair")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument(
        "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    controller_config['kp'] = 700 #TODO: meant for VR Controller

    # TODO: fix this Integration, dhanush

    #----Gaze Sensor Integration ----#
    gaze_client = SimpleClient('192.168.1.93', 5478, 96)
    gaze_client.connect_to_server()
    gaze_util_obj = gaze_data_util(3440, 1440)

    # --Format of Data -- #
    # (int(gaze_data_dict['pixel_x'])
    # int(gaze_data_dict['pixel_y']))
    #--------------------#


    
    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper_gaze(env, tmp_directory) # NOTE(dhanush) : USING A MODIFIED WRAPPER

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
    os.makedirs(new_dir)

    #VR Controller Code here
    policy = VRPolicy(
        max_lin_vel=1.0,
        max_rot_vel=0.5,
        pos_action_gain=2.0,
        rot_action_gain=0.5,
        rmat_reorder=[3, 1, 2,4]
    )

    # COLLECT DEMONSTRATIONS
    while True:
        collect_human_trajectory(policy, env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)

    
    # TERMINATING CONNECTION TO GAZE SOCKET
    gaze_client.disconnect()
