import torch

MAX_EPISODIC_REWARD = 100

class Config:
    ENV_NAME = "CartPole-v1"
    MAX_EPISODES = MAX_EPISODIC_REWARD
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter bounds
    HYPERPARAM_BOUNDS = {
        "lr": (1e-5, 1e-2),
        "epsilon_decay": (500.0, 5000.0),
        "batch_size": (32.0, 128.0),
    }
    LOG_KEYS = ["lr"]

    # Training params
    GAMMA = 0.99
    HIDDEN_DIM = 64
    REPLAY_CAPACITY = 5000
    TARGET_UPDATE_FREQ = 100