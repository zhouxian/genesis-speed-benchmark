import argparse
import time

import genesis as gs
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
                    
args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        # constraint_solver=gs.constraint_solver.Newton,
        # enable_self_collision=True,
        use_contact_island=True,
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

for i in range(10):
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3 * i, 0.0, 0.05),
        ),
    )
########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)


for i in range(100):
    scene.step()

t0 = time.perf_counter()
for i in range(500):
    scene.step()
t1 = time.perf_counter()

print(f'per env: {500 / (t1 - t0):,.2f} FPS')
print(f'total  : {500 / (t1 - t0) * n_envs:,.2f} FPS')