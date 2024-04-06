import argparse
import os
import random
import json
import h5py
import numpy as np
import imageio
import robosuite
from robosuite import make

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: 'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    )
    parser.add_argument("--use-actions", action="store_true")
    parser.add_argument("--video_path", type=str, default="video.mp4", help="Path to save the video")
    parser.add_argument("--skip_frame", type=int, default=1, help="Number of frames to skip before saving a new one")
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    demos = list(f["data"].keys())
    print(len(demos))

    
    # Initialize video writer
    # writer = imageio.get_writer(str(demos)+args.video_path, fps=20)

    while True:
        print("Playing back random episode... (press ESC to quit)")
        ep = random.choice(demos)

        video_file_name = ep + "_" + args.video_path
        writer = imageio.get_writer(video_file_name, fps=20)

        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = env.edit_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        states = f["data/{}/states".format(ep)][()]
        frame_count = 0

        if args.use_actions:
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]

            for j, action in enumerate(actions):
                env.step(action)
                env.render()

                # Save frame
                if frame_count % args.skip_frame == 0:
                    frame = env.sim.render(camera_name="agentview", width=512, height=512)
                    writer.append_data(frame)

                frame_count += 1

                if j < num_actions - 1:
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}")
        else:
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()

                # Save frame
                if frame_count % args.skip_frame == 0:
                    frame = env.sim.render(camera_name="agentview", width=512, height=512)
                    writer.append_data(frame)

                frame_count += 1

    writer.close() #
    f.close()
