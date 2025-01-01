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
        constraint_solver=gs.constraint_solver.CG, # to match mjx
        tolerance=1e-8, # to match mjx
    ),
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file="../../assets/xml/franka_emika_panda/mjx_panda_free.xml"),
)


########################## build ##########################
n_envs = args.B
scene.build(n_envs=n_envs)

# warmup
for i in range(200): # until touches the floor
    franka.control_dofs_position(np.array([0.33127472,  1.7633277 , -0.32836628, -0.23935018, -0.4920762 , 0.38396463,  0.39376438,  0.00876223,  0.03379252]))
    scene.step()

# NOTE: Although we use the same control signal here, visually the randomized motion in mjx is much smaller than in genesis (probably due to damping or some mismatch in controller), therefore, we expect mjx to be slower than the number measured here if the behavior is matched perfectly. (Feel free to reach out if this can be improved/corrected).

motor_dofs = np.arange(9)
position = franka.get_dofs_position()
t0 = time.perf_counter()
for i in range(1000):
    franka.control_dofs_position(position + torch.rand((n_envs, 9), device='cuda')*0.4 - 0.2, motor_dofs)
    scene.step()
t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')