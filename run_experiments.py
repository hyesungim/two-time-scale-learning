import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import time
import numpy as np
import pickle

from src.config import Config, MAX_EPISODIC_REWARD
from src.agent import DQNAgent
from src.pbt import generate_initial_population, pbt_step
from src.plotting import plot_deque_performance, plot_3d_evolution

def evaluate_worker(args):
    """Fixed worker: Trains for exactly steps_per_gen, ensuring sufficient experience."""
    agent, env_name, steps_per_gen = args

    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=100)

    agent.q_net.to(agent.device)
    agent.target_net.to(agent.device)

    state, _ = env.reset()
    current_ep_reward = 0.0

    gen_ep_rewards_sum = 0.0
    gen_ep_count = 0

    for _ in range(steps_per_gen):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train_step()
        agent.total_steps += 1

        current_ep_reward += reward

        if done:
            agent.recent_rewards.append(current_ep_reward)
            gen_ep_rewards_sum += current_ep_reward
            gen_ep_count += 1

            current_ep_reward = 0.0
            state, _ = env.reset()
        else:
            state = next_state

    env.close()

    agent.q_net.cpu()
    agent.target_net.cpu()

    fitness = np.mean(agent.recent_rewards) if len(agent.recent_rewards) > 0 else current_ep_reward
    raw_reward = (gen_ep_rewards_sum / gen_ep_count) if gen_ep_count > 0 else current_ep_reward

    return fitness, raw_reward, agent


def run_pbt_experiment(agents=None, pop_size=10, num_generations=10, steps_per_gen=500, seed=42, deque_size=2):
    print(f"\n{'=' * 70}")
    print(f"PBT EXPERIMENT: {pop_size} agents × {num_generations} generations ({steps_per_gen} steps/gen)")
    print(f"{'=' * 70}")

    env = gym.make(Config.ENV_NAME, max_episode_steps=MAX_EPISODIC_REWARD)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if agents is None:
        print("Initializing new population...")
        initial_hparams = generate_initial_population(pop_size, seed=seed)
        agents = [DQNAgent(state_dim, action_dim, h, deque_size=deque_size) for h in initial_hparams]

    history_fitness = []
    history_raw_rewards = []
    history_hparams = []
    detailed_ep_rewards = [[] for _ in range(pop_size)]

    ordered_keys = ["lr", "epsilon_decay", "batch_size"]
    hparams_array = np.zeros((num_generations + 1, pop_size, len(ordered_keys)))

    for a_idx, agent in enumerate(agents):
        hparams_array[0, a_idx, 0] = agent.hparams.get("lr", 0)
        hparams_array[0, a_idx, 1] = agent.hparams.get("epsilon_decay", 0)
        hparams_array[0, a_idx, 2] = agent.hparams.get("batch_size", 0)

    for gen in range(1, num_generations + 1):
        gen_start = time.time()

        fitness_scores = np.zeros(pop_size)
        raw_scores = np.zeros(pop_size)

        for i, ag in enumerate(agents):
            state, _ = env.reset()
            current_ep_reward = 0.0
            gen_ep_rewards_sum = 0.0
            gen_ep_count = 0

            for _ in range(steps_per_gen):
                action = ag.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                ag.replay_buffer.push(state, action, reward, next_state, done)
                ag.train_step()
                ag.total_steps += 1
                current_ep_reward += reward

                if done:
                    ag.recent_rewards.append(current_ep_reward)
                    gen_ep_rewards_sum += current_ep_reward
                    gen_ep_count += 1
                    detailed_ep_rewards[i].append((ag.total_steps, current_ep_reward))

                    current_ep_reward = 0.0
                    state, _ = env.reset()
                else:
                    state = next_state

            current_gen_avg = (gen_ep_rewards_sum / gen_ep_count) if gen_ep_count > 0 else current_ep_reward

            MAX_REWARD = MAX_EPISODIC_REWARD
            if len(ag.recent_rewards) == ag.recent_rewards.maxlen and current_gen_avg == MAX_REWARD:
                fitness_scores[i] = np.max(ag.recent_rewards)
            else:
                fitness_scores[i] = np.mean(ag.recent_rewards) if len(ag.recent_rewards) > 0 else current_ep_reward

            raw_scores[i] = current_gen_avg

            hparams_array[gen, i, 0] = ag.hparams.get("lr", 0)
            hparams_array[gen, i, 1] = ag.hparams.get("epsilon_decay", 0)
            hparams_array[gen, i, 2] = ag.hparams.get("batch_size", 0)

        history_fitness.append(fitness_scores.tolist())
        history_raw_rewards.append(raw_scores.tolist())
        history_hparams.append([ag.get_hparams() for ag in agents])

        gen_time = time.time() - gen_start
        print(
            f"Gen {gen:02d}/{num_generations} ({gen_time:.1f}s) | Avg Fit: {np.mean(fitness_scores):.1f} | Avg Raw: {np.mean(raw_scores):.1f}")

        if gen < num_generations:
            pbt_step(agents, fitness_scores, truncation_fraction=0.2)

    env.close()

    history_dict = {
        "fitness": history_fitness,
        "raw_rewards": history_raw_rewards,
        "hparams": history_hparams,
        "hparams_array": hparams_array,
        "detailed_ep_rewards": detailed_ep_rewards
    }

    return agents, history_dict


if __name__ == "__main__":
    k_values = [1, 2, 5]
    pop_size = 500
    num_gens = 10
    steps_per_gen = 500

    print(f"\n>>> STARTING BATCH EXPERIMENTS FOR DEQUE SIZES: {k_values} <<<")

    for k in k_values:
        print(f"\n\n{'#' * 50}")
        print(f"RUNNING EXPERIMENT FOR k = {k}")
        print(f"{'#' * 50}")

        agents, history_dict = run_pbt_experiment(
            pop_size=pop_size,
            num_generations=num_gens,
            steps_per_gen=steps_per_gen,
            seed=42,
            deque_size=k
        )

        filename = f"deque{k}_N{pop_size}_{steps_per_gen}steps.pkl"
        with open(filename, "wb") as f:
            pickle.dump(history_dict, f)

        print(f"Successfully saved {filename}")