import argparse
import os
import time

import cv2
import jax
import mujoco
import numpy as np
from mujoco import mjx

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=512) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
parser.add_argument("--c0", action="store_true", default=False) # disable collision
parser.add_argument("--c1", action="store_true", default=False) # enable collision but not self-collision
parser.add_argument("--mjxxml", action="store_true", default=False) # use panda xml designed for mjx (primitive collision)
parser.add_argument("-j", type=int, default=1) # jit steps
args = parser.parse_args()
                    

###### will boost performance by 30% according to mjx documentation ######
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags



n_envs = args.B
jit_steps = args.j

if args.mjxxml:
    mj_model = mujoco.MjModel.from_xml_path('../../assets/xml/franka_emika_panda/mjx_scene_free.xml')
else:
    mj_model = mujoco.MjModel.from_xml_path('../../assets/xml/franka_emika_panda/scene_free.xml')
mj_model.opt.timestep = 0.01

mj_model.opt.solver = 1 # CG

if args.c0: # disable collision
    mj_model.geom_contype = np.zeros_like(mj_model.geom_contype)
    mj_model.geom_conaffinity = np.zeros_like(mj_model.geom_conaffinity)
elif args.c1:
    if args.mjxxml: # mjxxml has no self-collision
        pass
    else:
        mj_model.geom_conaffinity[mj_model.geom_conaffinity == 1] = 2
        mj_model.geom_contype[0] = 2
        mj_model.geom_conaffinity[0] = 1

# Make model, data, and renderer
mj_data = mujoco.MjData(mj_model)
if args.v:
    renderer = mujoco.Renderer(mj_model, 480, 640)

mujoco.mj_resetData(mj_model, mj_data)

mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)
mjx_data = jax.vmap(lambda _: mjx_data)(np.arange(args.B))


def mjx_step_n(model, data):
    for _ in range(jit_steps):
        data = mjx.step(model, data)
    return data
jit_step_n = jax.jit(jax.vmap(mjx_step_n, in_axes=(None, 0)), backend='gpu')

# warmup
i_cur = 0
while True:
    mjx_data = jit_step_n(mjx_model, mjx_data)
    i_cur += jit_steps
    if i_cur >= 200:
        break
print('warmup done')

t0 = time.perf_counter()
i_cur = 0
while True:
    mjx_data = jit_step_n(mjx_model, mjx_data)

    if args.v:
        mj_data = mjx.get_data(mj_model, mjx_data)[0]
        renderer.update_scene(mj_data)
        img = renderer.render()

        cv2.imshow(f'mujoco', img[..., [2, 1, 0]])
        cv2.waitKey(1)

    i_cur += jit_steps
    if i_cur >= 1000:
        break

t1 = time.perf_counter()

print(f'per env: {1000 / (t1 - t0):,.2f} FPS')
print(f'total  : {1000 / (t1 - t0) * n_envs:,.2f} FPS')
