import os

import pybullet_envs
from pybullet_envs.stable_baselines.utils import TimeFeatureWrapper
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


env = make_vec_env("HopperBulletEnv-v0", n_envs=8, wrapper_class=TimeFeatureWrapper)

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., )

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="hopper_ppo",
            batch_size=128,
            n_steps=512,
            gamma=0.99, gae_lambda=0.92, alpha=99*2., beta=1*2.,
            n_epochs=20,
            ent_coef=0.0,
            sde_sample_freq=4,
            max_grad_norm=0.5,
            vf_coef=0.5,
            learning_rate=3e-5,
            use_sde=True,
            clip_range=0.4,
            policy_kwargs=dict(log_std_init=-2,
                               ortho_init=False,
                               activation_fn=nn.ReLU,
                               net_arch=[dict(pi=[256, 256], vf=[256, 256])]
                               ))

model.learn(total_timesteps=int(2e6), tb_log_name="ppo_exp_default", )

