"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Attractor
----------------
Positional control of franka panda robot with a target attractor that the robot tries to reach
"""

import math
import time

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil

n_envs = 1
n_envs = 512
# n_envs = 1024
# n_envs = 2048
# n_envs = 4096
# n_envs = 8192
# n_envs = 16384
# n_envs = 32768

vis = False
# vis = True

self_collision = True
# self_collision = False

free_drop = False
# free_drop = True

import torch

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="Franka Attractor Example")
# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.substeps = 1
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = True
if not vis:
    args.graphics_device_id = -1
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

if vis:
    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

# Add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# Load franka asset
asset_root = "../../assets"
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options)

# Set up the env grid
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)


print("Creating %d environments" % n_envs)
num_per_row = int(math.sqrt(n_envs))
envs = []
franka_handles = []
for i in range(n_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 0 if self_collision else 2)

    franka_handles.append(franka_handle)

if not free_drop:
    # get joint limits and ranges for Franka
    franka_dof_props = gym.get_actor_dof_properties(envs[0], franka_handles[0])
    franka_lower_limits = franka_dof_props['lower']
    franka_upper_limits = franka_dof_props['upper']
    franka_ranges = franka_upper_limits - franka_lower_limits
    franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
    franka_num_dofs = len(franka_dof_props)

    # override default stiffness and damping values
    franka_dof_props['stiffness'].fill(1000.0)
    franka_dof_props['damping'].fill(1000.0)

    # Give a desired pose for first 2 robot joints to improve stability
    franka_dof_props["driveMode"][0:2] = gymapi.DOF_MODE_POS

    franka_dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
    franka_dof_props['stiffness'][7:] = 1e10
    franka_dof_props['damping'][7:] = 1.0

    for i in range(n_envs):
        gym.set_actor_dof_properties(envs[i], franka_handles[i], franka_dof_props)

    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)

    dof_pos = dof_states[:, 0].view(n_envs, 9, 1)
    pos_action_ = torch.zeros_like(dof_pos).squeeze(-1).cuda()

if vis:
    # Point camera at environments
    cam_pos = gymapi.Vec3(-4.0, 4.0, -1.0)
    cam_target = gymapi.Vec3(0.0, 2.0, 1.0)

    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


pos_action = gymtorch.unwrap_tensor(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0.5, 0, 0.02, 0.02], device='cuda'),
        (n_envs, 1)
    )
)

gym.prepare_sim(sim)

def step():
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if not free_drop:
        gym.set_dof_position_target_tensor(sim, pos_action)

    # Step rendering
    if vis:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)

# warmup
for i in range(200):
    step()

t0 = time.perf_counter()
for i in range(1000):
    step()
t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')

if vis:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)