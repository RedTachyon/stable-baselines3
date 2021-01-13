import numpy as np


def get_beta_vector(T: int,
                    α: float,
                    β: float) -> np.ndarray:
    discount = np.zeros((1, T))

    current_discount = 1
    for t in range(T):
        discount[0, t] = current_discount
        current_discount *= (α + t) / (α + β + t)

    return discount


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

    # Process dones
    # !!! Assume all environments have episode ends simultaneously !!!
    steps_until_eoe = np.zeros((T,), dtype=int)
    is_final = np.zeros((T,), dtype=int)  # Might be starting too early OOBE
    counter = 0
    final = 1
    done = False
    for i, d in reversed(list(enumerate(dones[:, 0]))):
        if done:
            counter = 0
            done = False
            final = 0
        steps_until_eoe[i] = counter
        is_final[i] = final
        counter += 1
        done = d

    Γ = get_beta_vector(T + 1, α, β)
    lambdas = np.array([[λ ** l for l in range(T)]])

    #     γ = α / (α + β)
    #     Γ = np.array([[γ**l for l in range(T+1)]])

    factor = None

    for i in range(T):
        steps_left = steps_until_eoe[i]

        old_value = -values[i]
        future_rewards = (lambdas[:, :steps_left + 1] * Γ[:, :steps_left + 1]) @ rewards[i:i + steps_left + 1]

        if is_final[i]:
            steps_left += 1

            # Fix to properly handle the very last value of an episode
            if factor is None:
                factor = np.array([[1 - λ for i in range(steps_left)]]).T
                factor[-1] = 1.
            else:
                factor = factor[1:]

            future_values = (lambdas[:, :steps_left] * Γ[:, 1:steps_left + 1]) @ (
                        next_values[i:i + steps_left] * next_non_terminal[i:i + steps_left] * factor[-steps_left:])

        else:
            future_values = (1 - λ) * (lambdas[:, :steps_left] * Γ[:, 1:steps_left + 1]) @ (
                        next_values[i:i + steps_left] * next_non_terminal[i:i + steps_left])

        advantages[i] = old_value + future_rewards + future_values

    returns = advantages + values

    return returns, advantages


