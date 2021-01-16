import numpy as np
from numba import njit
from typing import Tuple


def convert_params(mu: float, eta: float) -> Tuple[float, float]:
    """
    Convert the mu-eta parametrization to alpha-beta
    """
    alpha = mu / (eta * (mu - 1))
    beta = 1./eta

    return alpha, beta


def heuritic_params(mu: float) -> Tuple[float, float]:
    p = (1 - mu) / mu
    sigma = 1 - mu

    a = (p - sigma ** 2) / sigma ** 2
    b = p * a

    return 2 * a, 2 * b


@njit
def get_beta_vector(T: int,
                    α: float,
                    β: float) -> np.ndarray:
    discount = np.zeros((1, T), dtype=np.float32)

    current_discount = 1
    for t in range(T):
        discount[0, t] = current_discount
        current_discount *= (α + t) / (α + β + t)

    return discount


@njit
def beta_gae(rewards: np.ndarray,  # [T, N]
             values: np.ndarray,  # [T, N]
             last_values: np.ndarray,  # [1, N]
             dones: np.ndarray,  # [T, N], actually next_dones (whether previous step was terminal)
             final_dones: np.ndarray,  # [1, ]
             α: float = 99.,
             β: float = 1.,
             λ: float = 0.95):
    T = rewards.shape[0]
    N = rewards.shape[1]  # Number of envs

    advantages = np.zeros((T, N), dtype=np.float32)

    final_dones = final_dones.reshape((1, N))
    last_values = last_values.reshape((1, N))
    next_non_terminal = 1 - np.concatenate((dones, final_dones))[1:]
    next_values = np.concatenate((values, last_values))[1:]

    Γ = get_beta_vector(T + 1, α, β)
    lambdas = np.array([[λ ** l for l in range(T)]], dtype=np.float32)

    #     γ = α / (α + β)
    #     Γ = np.array([[γ**l for l in range(T+1)]])

    for n in range(N):
        # Done preprocessing step
        steps_until_eoe = np.zeros((T,), dtype=np.int32)
        is_final = np.zeros((T,), dtype=np.int32)
        counter = 0
        final = 1
        done = False
        factor = None

        for i, d in list(enumerate(dones[:, n]))[::-1]:
            if done:
                counter = 0
                done = False
                final = 0
            steps_until_eoe[i] = counter
            is_final[i] = final
            counter += 1
            done = d

        for i in range(T):
            steps_left = steps_until_eoe[i]

            old_value = -values[i, n]
            future_rewards = (lambdas[:, :steps_left + 1] * Γ[:, :steps_left + 1]) @ rewards[i:i + steps_left + 1,
                                                                                     n:n + 1]

            if is_final[i]:
                steps_left += 1

                # Fix to properly handle the very last value of an episode
                if factor is None:
                    factor = np.array([[1 - λ for i in range(steps_left)]], dtype=np.float32).T
                    factor[-1] = 1.
                else:
                    factor = factor[1:]

                future_values = (lambdas[:, :steps_left] * Γ[:, 1:steps_left + 1]) @ (
                        next_values[i:i + steps_left, n:n + 1] * next_non_terminal[i:i + steps_left, n:n + 1] * factor[
                                                                                                                -steps_left:])

            else:
                future_values = np.float32(1. - λ) * (lambdas[:, :steps_left] * Γ[:, 1:steps_left + 1]) @ (
                        next_values[i:i + steps_left, n:n + 1] * next_non_terminal[i:i + steps_left, n:n + 1])

            total = old_value + future_rewards + future_values
            advantages[i, n] = total[0, 0]

    returns = advantages + values

    return returns, advantages


