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


class Parser(BaseParser):
    directory: Optional[str]
    num_experiments: int
    num_workers: int
    gammas: List[float]
    etas: Optional[List[float]]

    beta: bool
    gamma_range: bool
    eta_range: bool
    eta_invert: bool

    _abbrev = {
        "directory": "d",
        "num_experiments": "n",
        "num_workers": "w",
        "gammas": "g",
        "etas": "e",
        "beta": "b",
        "gamma_range": "gr",
        "eta_range": "er",
        "eta_invert": "ei"
    }

    _default = {
        "directory": "hopper_ppo",
        "num_experiments": 5,
        "num_workers": 5,
    }

    _help = {
        "directory": "Directory of the tensorboard logs",
        "num_experiments": "How many random seeds to run",
        "gammas": "Values of gamma (or mu) to test",
        "etas": "If in beta mode, values of eta (1/beta) to test",
        "beta": "Whether to use beta-weighted discounting. In absence of eta values, the heuristic will be used",
        "gamma_range": "Whether the gamma values should be interpreted as in np.linspace(start, end, num)",
        "eta_range": "See gamma_range",
        "eta_invert": "Whether each value of eta should be inverted - this is used to do a gridsearch on"
                      " beta values instead, putting more density on lower values of eta"
    }


def run_experiment(i: int,  # index
                   total: int,
                   gamma: float,  # gamma or mu
                   eta: Optional[float] = None,  # eta = 1/beta
                   dir_name: str = "hopper_ppo",  # TB directory
                   use_heuristic: bool = False):  # use beta discount

    if eta:
        alpha, beta = convert_params(gamma, eta)
        params = {
            "gamma": None,
            "alpha": alpha,
            "beta": beta,
        }
        param_string = f"{gamma:.3f}mu_{round(beta)}beta"
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

    print(f"Experiment {i}/{total}: {exp_name}")

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


def dummy_task(*args):
    time.sleep(1)
    print(args)


if __name__ == '__main__':
    args = Parser()

    if args.gamma_range:
        g_min, g_max, g_num = args.gammas
        g_num = round(g_num)  # int

        gammas = np.linspace(g_min, g_max, g_num)
    else:
        gammas = np.array(args.gammas)

    if args.eta_range:
        e_min, e_max, e_num = args.etas
        e_num = round(e_num)  # int

        etas = np.linspace(e_min, e_max, e_num)
    else:
        etas = np.array(args.etas)

    if etas == None:
        etas = [None]
    elif args.eta_invert:
        etas = 1./etas

    workers = args.num_workers

    args = [
        (gamma, eta, args.directory, args.beta)
        for _ in range(args.num_experiments)
        for gamma in gammas
        for eta in etas
    ]
    args = [(i, len(args), *arg) for i, arg in enumerate(args)]

    print(f"Gamma values: {gammas}")
    print(f"Eta values: {etas}")

    print(f"Running {len(args)} experiments on {workers} workers")

    with mp.Pool(workers) as pool:
        pool.starmap(run_experiment, args)
