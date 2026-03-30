import numpy as np
import random
import copy
from src.config import Config

def perturb_hparams(hparams, mutation_strength=0.1):
    new = hparams.copy()
    for key in new.keys():
        low, high = Config.HYPERPARAM_BOUNDS[key]
        val = new[key]
        is_log = key in Config.LOG_KEYS

        if is_log:
            log_low, log_high = np.log10(low), np.log10(high)
            log_val = np.log10(val)
            norm_val = 2.0 * (log_val - log_low) / (log_high - log_low) - 1.0
        else:
            norm_val = 2.0 * (val - low) / (high - low) - 1.0

        norm_val += mutation_strength * np.random.normal(0, 1)
        norm_val = np.clip(norm_val, -1.0, 1.0)

        if is_log:
            log_new = (norm_val + 1.0) / 2.0 * (log_high - log_low) + log_low
            new_val = float(10 ** log_new)
        else:
            new_val = (norm_val + 1.0) / 2.0 * (high - low) + low

        if key == "batch_size":
            new_val = int(round(new_val))
        else:
            new_val = float(new_val)
        new[key] = new_val
    return new

def pbt_step(agents, fitness_scores, truncation_fraction=0.2):
    pop_size = len(agents)
    scores = np.array(fitness_scores)
    k = max(1, int(np.ceil(pop_size * truncation_fraction)))

    sorted_indices = np.argsort(scores)
    bottom_indices = sorted_indices[:k]
    top_indices = sorted_indices[-k:]

    for i in bottom_indices:
        donor_idx = np.random.choice(top_indices)
        donor = agents[donor_idx]

        weights = copy.deepcopy(donor.q_net.state_dict())
        new_hparams = perturb_hparams(donor.get_hparams())

        agents[i].load_weights(weights)
        agents[i].set_hparams(new_hparams)
        agents[i].replay_buffer.clear()
        agents[i].recent_rewards.clear()

def generate_initial_population(pop_size, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    initial_hparams = []
    for _ in range(pop_size):
        hparams = {
            "lr": 10 ** random.uniform(-5, -2),
            "epsilon_decay": random.uniform(500, 5000),
            "batch_size": random.choice([32, 64, 128])
        }
        initial_hparams.append(hparams)
    return initial_hparams