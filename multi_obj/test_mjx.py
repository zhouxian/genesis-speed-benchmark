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
parser.add_argument("-n", type=int, default=10) # n cubes
parser.add_argument("-v", action="store_true", default=False) # visualize
args = parser.parse_args()
                    

###### will boost performance by 30% according to mjx documentation ######
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags



n_envs = args.B
jit_steps = 1

xml = f"""
<mujoco>

    <statistic center="0.3 0 0.4" extent="1"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>

    <asset>

        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
"""
np.random.seed(0)
for i in range(args.n):
    pos = np.array([0.2*i, 0.0, 0.05])

    xml += f'''
        <body name="obj_{i}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}">
            <freejoint/>
            <geom name="obj_{i}" type="box" size="0.05 0.05 0.05" contype="1" conaffinity="1"/>
        </body>
    '''

xml += """
    </worldbody>
</mujoco>
"""

mj_model = mujoco.MjModel.from_xml_string(xml)

mj_model.opt.solver = 1 # CG
mj_model.opt.timestep = 0.01



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
