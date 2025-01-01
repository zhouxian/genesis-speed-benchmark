import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("-r", action="store_true", default=False) # random action
                    
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

obj = scene.add_entity(
    morph=gs.morphs.URDF(
        file="../assets/urdf/bottle/bottle.urdf",
        scale=0.09,
        pos=(0.65, 0.0, 0.036),
        euler=(0, 90, 0),
    ),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="../assets/xml/franka_emika_panda/panda.xml")
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

grasp_qpos = np.array([-0.9937,  1.4588,  1.3058, -1.6924, -1.4882,  1.8461,  1.4577,  0.04, 0.04])
lift_qpos = np.array([-1.0411,  1.2861,  1.5200, -1.7065, -1.2946,  1.6572,  1.4315,  0.04, 0.04])
franka.set_dofs_position(grasp_qpos)

# grasp
franka.control_dofs_position(grasp_qpos[:-2], motors_dof)
franka.control_dofs_force(np.array([-4, -4]), fingers_dof)


for i in range(100):
    scene.step()

# lift
franka.control_dofs_position(lift_qpos[:-2], motors_dof)
for i in range(50):
    scene.step()

ref_pos = torch.tile(torch.tensor(lift_qpos[:7]), [n_envs, 1]).cuda()

t0 = time.perf_counter()
for i in range(500):
    if args.r and i % 2 == 0: # match maniskill's 50hz control freq
        # match maniskill's random action
        franka.control_dofs_position(ref_pos + torch.rand((n_envs, 7), device='cuda')*0.05 - 0.025, motors_dof)
    scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')