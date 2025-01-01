# mostly identical to https://gist.github.com/StoneT2000/904b2f8fd8cd9d4ed43b9772ee044c1d

import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("-c", action="store_true", default=False) # self_collision -> turning this on gives only ~5% speed drop
parser.add_argument("-r", action="store_true", default=False) # random action
                    
args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        enable_self_collision=args.c,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.URDF(file="assets/urdf/franka_description/robots/franka_panda.urdf", fixed=True),
)

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)


franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 0.5, 0, 0.02, 0.02], device=gs.device), (n_envs, 1)),
)


# warmup
for i in range(200):
    scene.step()

motor_dofs = np.arange(9)
position = franka.get_dofs_position()

t0 = time.perf_counter()
for i in range(1000):
    if args.r:
        franka.control_dofs_position(position + torch.rand((n_envs, 9), device='cuda')*0.02, motor_dofs)
    scene.step()
t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')