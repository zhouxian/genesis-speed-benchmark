import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-r", action="store_true", default=False) # randomize action
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("--mjx-solver-setting", action="store_true", default=False)

args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
rigid_options = gs.options.RigidOptions(
    dt=0.01,
    constraint_solver = gs.constraint_solver.Newton,
)

if args.mjx_solver_setting: # use solver setting suggested by mujoco official go2 xml (https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_go2/go2_mjx.xml)
    rigid_options.tolerance     = 1e-8
    rigid_options.iterations    = 1
    rigid_options.ls_iterations = 5
    
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=rigid_options,
)

########################## entities ##########################
scene.add_entity(
    gs.morphs.Plane(),
)
robot = scene.add_entity(
    gs.morphs.URDF(
        file="../../assets/urdf/go2/urdf/go2.urdf",
        pos=(0, 0, 0.8),
    ),
)
########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

joint_names = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

motor_dofs = np.array([robot.get_joint(name).dof_idx_local for name in joint_names])

# match isaacgym
robot.set_dofs_kp(np.full(12, 1000), motor_dofs)
robot.set_dofs_kv(np.full(12, 10), motor_dofs)

default_qpos = torch.tile(
    torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5],
        device='cuda'
    ),
    (n_envs, 1)
)
robot.control_dofs_position(
    default_qpos,
    motor_dofs
)

# warmup
for i in range(200):
    scene.step()

if args.r:
    t0 = time.perf_counter()
    for i in range(1000):
        # random action
        robot.control_dofs_position(default_qpos + torch.rand((n_envs, 12), device='cuda')*0.4-0.2, motor_dofs)
        scene.step()
    t1 = time.perf_counter()

else:
    t0 = time.perf_counter()
    for i in range(1000):
        scene.step()
    t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')