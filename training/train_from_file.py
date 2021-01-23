from pybullet_envs.stable_baselines.utils import TimeFeatureWrapper
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.beta_discount import convert_params, heuritic_params

from typarse import BaseParser
from typing import List, Optional
import numpy as np
import multiprocessing as mp
import time
import yaml

from tqdm import tqdm

class Parser(BaseParser):
    directory: Optional[str]
    num_workers: int

    config: str
    config_name: str

    beta: bool
    gamma_range: bool
    eta_range: bool
    eta_invert: bool

    _abbrev = {
        "directory": "d",
        "num_workers": "w",
        "eta_invert": "ei",
        "config": "c",
        "config_name": "cn",
    }

    _default = {
        "directory": "hopper_ppo",
        "num_workers": 5,
    }

    _help = {
        "directory": "Directory of the tensorboard logs",
        "gamma_range": "Whether the gamma values should be interpreted as in np.linspace(start, end, num)",
        "eta_range": "See gamma_range",
        "eta_invert": "Whether each value of eta should be inverted - this is used to do a gridsearch on"
                      " beta values instead, putting more density on lower values of eta"
    }


def run_experiment(args):  # use beta discount
    i: int
    gamma: float
    eta: Optional[float]  # eta = 1/beta
    dir_name: str  # TB directory
    use_heuristic: bool  # use beta discount
    eta_invert: bool

    i, gamma, eta, dir_name, use_heuristic, eta_invert = args

    if eta:
        alpha, beta = convert_params(gamma, eta)
        params = {
            "gamma": None,
            "alpha": alpha,
            "beta": beta,
        }
        if eta_invert:
            param_string = f"{gamma:.3f}mu_{beta:.3f}beta"
        else:
            param_string = f"{gamma:.3f}mu_{eta:.3f}eta"
    elif use_heuristic:
        alpha, beta = heuritic_params(gamma)
        params = {
            "gamma": None,
            "alpha": alpha,
            "beta": beta
        }
        param_string = f"{gamma:.3f}mu_heuristic"
    else:
        params = {
            "gamma": gamma
        }
        param_string = f"{gamma:.3f}g"

    exp_name = f"ppo_{'beta' if eta else 'exp'}_{param_string}"
    # time.sleep(1)
    # print(f"\b\r \rExperiment {i}: {exp_name}")
    # return i, exp_name
    env = make_vec_env("HopperBulletEnv-v0", n_envs=16, wrapper_class=TimeFeatureWrapper)

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)


    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=dir_name,
                batch_size=128,
                n_steps=512,
                gae_lambda=0.92,
                **params,  # gamma, alpha, beta
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

    model.learn(total_timesteps=int(2e6), tb_log_name=exp_name)
    return i, exp_name


def dummy_task(*args):
    time.sleep(1)
    print(args)


if __name__ == '__main__':
    # mp.set_start_method('spawn')

    args = Parser()
    workers = args.num_workers

    with open(args.config, "r") as f:
        config = yaml.load(f)

    params = config[args.config_name]
    if args.eta_invert:
        params = [(gamma, 1./eta) for gamma, eta in params]

    args = [
        (gamma, eta, args.directory, args.beta, args.eta_invert)
        for gamma, eta in params
    ]

    args = [(i, *arg) for i, arg in enumerate(args)]

    print(f"Running {len(args)} experiments on {workers} workers from a file")

    with mp.get_context("spawn").Pool(workers) as pool:
        with tqdm(total=len(args), desc="Starting training") as pbar:
            for j, (i, exp_name) in enumerate(pool.imap_unordered(run_experiment, args)):
                pbar.set_description(desc=f"Last started: {exp_name}")
                pbar.update()
