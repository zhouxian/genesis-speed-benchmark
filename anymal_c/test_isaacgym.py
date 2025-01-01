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
n_envs = 4096
# n_envs = 8192
# n_envs = 16384
# n_envs = 32768

vis = False
# vis = True

random_action = False
random_action = True

spacing = 1.0

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments()
# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.substeps = 1
sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
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
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load franka asset
asset_root = "../assets/"
franka_asset_file = "urdf/anymal_c/urdf/anymal_c.urdf"

asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.armature = 0.01
print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options)

franka_dof_props = gym.get_asset_dof_properties(franka_asset)

franka_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][:].fill(1000.0)
franka_dof_props["damping"][:].fill(10.0)
    
# Set up the env grid
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
pose = gymapi.Transform()
pose.r = gymapi.Quat(0, 0, 0, 1)
pose.p.z = 1.
pose.p.x = 5
pose.p.y = 5


# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

print("Creating %d environments" % n_envs)
num_per_row = int(math.sqrt(n_envs))
envs = []
franka_handles = []
for i in range(n_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 2) # disable self-collision by default
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    franka_handles.append(franka_handle)

# Point camera at environments
cam_pos = gymapi.Vec3(4, -4.0, 4.0)
cam_target = gymapi.Vec3(5.5, 5.5, 1.0)

if vis:
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)



_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
import torch

dof_pos = dof_states[:, 0].view(n_envs, 12, 1)
pos_action_ = torch.zeros_like(dof_pos).squeeze(-1).cuda()

gym.prepare_sim(sim)
pos_action = gymtorch.unwrap_tensor(torch.zeros([n_envs, 12], device='cuda'))
gym.set_dof_position_target_tensor(sim, pos_action)

def step():
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    if random_action:
        pos_action = gymtorch.unwrap_tensor(torch.rand((n_envs, 12), device='cuda')*0.4 - 0.2)
        gym.set_dof_position_target_tensor(sim, pos_action)

    # Step rendering
    if vis:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)


# warmup
for i in range(200):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

t0 = time.perf_counter()
for i in range(1000):
    step()
t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')

if vis:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)