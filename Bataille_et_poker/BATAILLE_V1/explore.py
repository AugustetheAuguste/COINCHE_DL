import tkinter as tk
import numpy as np
import gym
from stable_baselines3 import PPO

# Même définition de l'environnement que pour l'entraînement
class CardGameEnv(gym.Env):
    """
    Environnement simplifié du jeu de carte.
    Voir les commentaires dans train.py pour les détails.
    """
    def __init__(self, episode_length=100):
        super(CardGameEnv, self).__init__()
        self.episode_length = episode_length
        self.current_step = 0
        self.observation_space = gym.spaces.Box(low=1, high=13, shape=(1,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(52)
    
    def reset(self):
        self.current_step = 0
        self.table_card = np.random.randint(1, 14)
        return np.array([self.table_card], dtype=np.int32)
    
    def step(self, action):
        played_rank = (action // 4) + 1
        if played_rank > self.table_card:
            reward = 1
        elif played_rank == self.table_card:
            reward = 0
        else:
            reward = -1
        info = {"table_card": self.table_card, "played_rank": played_rank}
        self.current_step += 1
        done = self.current_step >= self.episode_length
        self.table_card = np.random.randint(1, 14)
        return np.array([self.table_card], dtype=np.int32), reward, done, info

def main():
    # Initialisation de l'environnement
    env = CardGameEnv()
    obs = env.reset()

    # Chargement du meilleur modèle sauvegardé
    model_path = "./best_model/best_model.zip"
    model = PPO.load(model_path)

    # Création de l'interface graphique
    root = tk.Tk()
    root.title("Exploration de l'Agent - Jeu de Cartes")
    root.geometry("400x300")

    # Création des labels pour afficher les informations
    table_label = tk.Label(root, text="Carte sur la table : ?", font=("Helvetica", 20))
    table_label.pack(pady=10)
    agent_label = tk.Label(root, text="Carte jouée par l'agent : ?", font=("Helvetica", 20))
    agent_label.pack(pady=10)
    reward_label = tk.Label(root, text="Reward : ?", font=("Helvetica", 20))
    reward_label.pack(pady=10)

    # Fonction pour passer à la manche suivante
    def next_round():
        nonlocal obs
        # L'agent prédit l'action à partir de l'observation courante
        action, _ = model.predict(obs, deterministic=True)
        new_obs, reward, done, info = env.step(action)
        # Affiche la carte qui était sur la table pour cette manche
        table_label.config(text=f"Carte sur la table : {info['table_card']}")
        agent_label.config(text=f"Carte jouée par l'agent : {info['played_rank']}")
        reward_label.config(text=f"Reward : {reward}")
        obs = new_obs
        if done:
            obs = env.reset()

    # Bouton pour lancer la manche suivante
    next_button = tk.Button(root, text="Next Round", command=next_round, font=("Helvetica", 16))
    next_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
