import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from modele_jeu_v2 import CoincheEnv
from stable_baselines3.common.callbacks import BaseCallback


class MetricsLogger(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.losses = []
        self.mean_rewards = []
        self.timesteps = []

    def _on_step(self):
        # Enregistrement des récompenses moyennes
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    ep_reward += reward
                rewards.append(ep_reward)
            mean_reward = np.mean(rewards)
            self.mean_rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
            print(f"[Step {self.num_timesteps}] ➤ Reward moyen : {mean_reward:.2f}")

        # Récupérer la loss depuis le logger (optionnel)
        if 'train/loss' in self.model.logger.name_to_value:
            loss = self.model.logger.name_to_value['train/loss']
            self.losses.append(loss)

        return True


def train_agent():
    """Entraîne l'IA à jouer à la Coinche avec un LSTM."""
    env = CoincheEnv()
    eval_env = CoincheEnv()

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_coinche_logs",
        gamma=0.99,
        n_steps=512,
        batch_size=128,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.1,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": [256, 256],
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": False
        },
    )

    print("Démarrage de l'entraînement...")

    # Callback pour enregistrement
    metrics_callback = MetricsLogger(eval_env, eval_freq=10,n_eval_episodes = 3)
    model.learn(total_timesteps=100, tb_log_name="PPO_1", callback=metrics_callback, log_interval=None)
    model.save("coinche_ia_v2")

    print("Entraînement terminé !")

    # Sauvegarde des métriques
    np.save("losses.npy", np.array(metrics_callback.losses))
    np.save("rewards.npy", np.array(metrics_callback.mean_rewards))
    np.save("timesteps.npy", np.array(metrics_callback.timesteps))


def plot_metrics():
    """Affiche les courbes de récompense et de loss."""
    losses = np.load("losses.npy", allow_pickle=True)
    rewards = np.load("rewards.npy", allow_pickle=True)
    timesteps = np.load("timesteps.npy", allow_pickle=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(timesteps[:len(rewards)], rewards, label='Reward')
    plt.title("Récompense moyenne")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='orange')
    plt.title("Loss au cours de l'entraînement")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_agent()
    plot_metrics()
