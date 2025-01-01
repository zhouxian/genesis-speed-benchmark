import argparse
import os
import time

import cv2
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

parser = argparse.ArgumentParser()
parser.add_argument("-B", type=int, default=512) # batch size
parser.add_argument("-v", action="store_true", default=False) # visualize
args = parser.parse_args()
                    

###### will boost performance by 30% according to mjx documentation ######
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


n_envs = args.B
jit_steps = 1

mj_model = mujoco.MjModel.from_xml_path('../../assets/xml/franka_emika_panda/mjx_scene.xml')
mj_model.opt.timestep = 0.01
mj_model.opt.solver = 1 # CG

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

# copied from free fall script, qpos when touching ground
ref_pos = jnp.tile(jnp.array([ 0.33127472,  1.7633277 , -0.32836628, -0.23935018, -0.4920762 , 0.38396463,  0.39376438,  0.00876223,  0.03379252]), [n_envs, 1])

mjx_data = mjx_data.replace(ctrl = ref_pos)
# warmup
i_cur = 0
while True:
    mjx_data = jit_step_n(mjx_model, mjx_data)
    i_cur += jit_steps
    if i_cur >= 200: # until touches the floor
        break
print('warmup done')

key = jax.random.PRNGKey(0)

# NOTE: Although we use the same control signal here, visually the randomized motion in mjx is much smaller than in genesis (probably due to damping or some mismatch in controller); therefore, we expect mjx to be further slower than the number measured here if the behavior is matched perfectly. (Feel free to reach out if this can be improved/corrected).

t0 = time.perf_counter()
i_cur = 0
while True:
    mjx_data = mjx_data.replace(ctrl = ref_pos + jax.random.uniform(key, shape=(n_envs, 9)) * 0.4 - 0.2)
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
