import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("-r", action="store_true", default=False) # random action
parser.add_argument("-m", action="store_true", default=False) # move along a traj
parser.add_argument("--mjcf", action="store_true", default=False) # use mjcf
                    
args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.Newton,
        enable_self_collision=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    ),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml") if args.mjcf else gs.morphs.URDF(file="../assets/urdf/franka_description/robots/franka_panda.urdf", fixed=True),
)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# set control gains (official value taken from mujoco franka xml)
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

grasp_qpos = np.array([-1.0104,  1.5623,  1.3601, -1.6840, -1.5863,  1.7810,  1.4598,  0.0400, 0.0400])
lift_qpos = np.array([-1.0426,  1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510,  0., 0.])
franka.set_dofs_position(grasp_qpos)

# grasp
franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)


for i in range(100):
    scene.step()

# lift
franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for i in range(50):
    scene.step()

ref_pos = torch.tile(torch.tensor(lift_qpos[:7]), [n_envs, 1]).cuda()

t0 = time.perf_counter()
if args.m:
    init_qpos = torch.tile(torch.tensor([-1.0426,  1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510]), [n_envs, 1]).cuda()
    target_qpos = torch.tile(torch.tensor([1.0426,  -1.4028,  1.5634, -1.7114, -1.4055,  1.6015,  1.4510]), [n_envs, 1]).cuda()
    for i in range(500):
        if i % 2 == 0: # match maniskill's 50hz control freq
            franka.control_dofs_position(init_qpos + (target_qpos - init_qpos) * i / 500, motors_dof)
        scene.step()
else:
    for i in range(500):
        if args.r and i % 2 == 0: # match maniskill's 50hz control freq
            # match maniskill's random action
            franka.control_dofs_position(ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025, motors_dof)
        scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')