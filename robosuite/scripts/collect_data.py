"""
A script to collect a batch of human demonstrations.

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

import robosuite.utils.transform_utils as T

import robosuite as suite
import robosuite.macros as macros
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from copy import deepcopy

# Global var for linking pybullet server to multiple ik controller instances if necessary
# pybullet_server = None
def collect_human_trajectory(policy, qpos, env, device, arm, env_configuration):
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
    env.robots[0].init_qpos = qpos
    env.robots[0].reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]
        
        # import ipdb; ipdb.set_trace()
        cart_pos = np.concatenate((active_robot.controller.ee_pos, T.mat2euler(active_robot.controller.ee_ori_mat)))
        obs_dict = {
            "robot_state": {
                "cartesian_position": cart_pos,
                # "gripper_position": 0
            }
        }
        action = policy.forward(obs_dict)
        # # Get the newest action
        # action, grasp = input2action(
        #     device=device, robot=active_robot, active_arm=arm, env_configuration=env_configuration
        # )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # Run environment step
        env.step(action)
        
        # clip action
        action = np.clip(action, -1, 1)
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


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
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
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
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

            num_eps += 1
            ep_data_grp = grp.create_group("demo_{}".format(num_eps))

            # store model xml as an attribute
            xml_path = os.path.join(directory, ep_directory, "model.xml")
            with open(xml_path, "r") as f:
                xml_str = f.read()
            ep_data_grp.attrs["model_file"] = xml_str

            # write datasets for states and actions
            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))
        else:
            print("Demonstration is unsuccessful and has NOT been saved")

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    # grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) 
    import robomimic.envs.env_base as EB

    env_meta = dict(
        type=EB.EnvType.ROBOSUITE_TYPE,
        env_name=env_name,
        # env_version=f["data"].attrs["repository_version"],
        env_kwargs=env_info,
    )
    grp.attrs["env_args"] = json.dumps(env_meta, indent=4)

    f.close()


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
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
    # import ipdb; ipdb.set_trace()

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

    ik_controller_config = load_controller_config(default_controller="IK_POSE")
    ik_controller_config["robot_name"] = env.robots[0].name
    ik_controller_config["sim"] = env.robots[0].sim
    ik_controller_config["eef_name"] = env.robots[0].gripper.important_sites["grip_site"]
    ik_controller_config["eef_rot_offset"] = env.robots[0].eef_rot_offset
    ik_controller_config["joint_indexes"] = {
        "joints": env.robots[0].joint_indexes,
        "qpos": env.robots[0]._ref_joint_pos_indexes,
        "qvel": env.robots[0]._ref_joint_vel_indexes,
    }
    ik_controller_config["actuator_range"] = env.robots[0].torque_limits
    ik_controller_config["policy_freq"] = env.robots[0].control_freq
    ik_controller_config["ndim"] = len(env.robots[0].robot_joints)

    ori_interpolator = None
    interpolator = None
    if interpolator is not None:
        interpolator.set_states(dim=3)  # EE IK control uses dim 3 for pos and dim 4 for ori
        ori_interpolator = deepcopy(interpolator)
        ori_interpolator.set_states(dim=4, ori="quat")
    
    from robosuite.controllers import controller_factory
    ik = controller_factory(name="IK_POSE", params=ik_controller_config)
    # Import pybullet server if necessary
    # import ipdb; ipdb.set_trace()
    ik_pose = ik.ik_robot_eef_joint_cartesian_pose()

    qposs = []
    for _ in range(200):
        offset = np.random.uniform(-0.05, 0.05, size=3)
        new_pos = ik_pose[0] + offset
        new_pos[0] += 0.1 # moves it forward 
        new_pos[1] += 0.1 # moves it to the left

        # rotate z by 90 degrees
        rotation = T.axisangle2quat(np.array([0,0,np.radians(45)]))
        
        # sample a random rotation vector around z axis 
        # rotation = T.axisangle2quat(np.array([0,0,np.random.uniform(-np.pi / 2, np.pi / 2, size=1)]))
        new_quat = T.quat_multiply(rotation, ik_pose[1])

        new_ik_pose = (new_pos, new_quat)
        qpos = ik.inverse_kinematics(
            target_position=new_ik_pose[0], target_orientation=new_ik_pose[1]
        )
        qposs.append(qpos)

    # Map input to quaternion
    # rotate z by 90 degrees
    # rotation = T.axisangle2quat(np.array([0,0,np.radians(45)]))
    # quat = T.quat_multiply(rotation, ik_pose[1])

    # ik_pose = (ik_pose[0], quat)

    # # pose_world = ik.bullet_base_pose_to_world_pose(ik_pose)
    # pose_world = ik_pose
    # qpos = ik.inverse_kinematics(
    #    target_position=pose_world[0], target_orientation=pose_world[1]
    # )
    # print(qpos)
    # env.robots[0].init_qpos = qpos
    # env.robots[0].reset()

    # global pybullet_server
    # from robosuite.controllers.ik import InverseKinematicsController

    # # if pybullet_server is None:
    # from robosuite.controllers.ik import PyBulletServer
    # from robosuite.controllers.controller_factory import get_pybullet_server, reset_controllers

    # # pybullet_server = get_pybullet_server()
    # # pybullet_server = Py
    # pybullet_server = PyBulletServer()

    # reset_controllers()
    # print(get_pybullet_server())
    # ik = InverseKinematicsController(
    #     interpolator_pos=interpolator,
    #     interpolator_ori=ori_interpolator,
    #     bullet_server_id=pybullet_server.server_id,
    #     **ik_controller_config,
    # )
    # import ipdb; ipdb.set_trace()

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

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


    # reset to a new configuration near the object
    # ik_pose = ik.ik_robot_eef_joint_cartesian_pose()

    # Map input to quaternion
    # sample some random offset from the current position
    from robosuite.utils.oculus_controller import VRPolicy
    policy = VRPolicy(
        max_lin_vel=1.0,
        max_rot_vel=0.5,
        pos_action_gain=2.0,
        rot_action_gain=0.5,
        rmat_reorder=[3, 1, 2,4]
    )
    count = 0 
    # collect demonstrations
    while True:
        # env.robots[0].init_qpos = qpos
        # env.robots[0].reset()
        collect_human_trajectory(policy, qposs[count], env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
        count += 1