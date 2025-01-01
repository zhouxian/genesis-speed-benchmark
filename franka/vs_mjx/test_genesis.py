import argparse
import time

import genesis as gs
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=1) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("-c", action="store_true", default=False) # enable collision but not self-collision (same as mjx)
parser.add_argument("--mjxxml", action="store_true", default=False) # use panda xml designed for mjx (primitive collision, without self-collision)
                    
args = parser.parse_args()

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    show_viewer=args.v,
    rigid_options=gs.options.RigidOptions(
        dt=0.01,
        constraint_solver=gs.constraint_solver.CG, # to match mjx
        enable_collision=args.c,
        tolerance=1e-8, # to match mjx
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

if args.mjxxml:
    franka = scene.add_entity(
        gs.morphs.MJCF(file="../../assets/xml/franka_emika_panda/mjx_panda_free.xml"),
    )
else:
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# warmup
for i in range(200):
    scene.step()

t0 = time.perf_counter()
for i in range(1000):
    scene.step()
t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')