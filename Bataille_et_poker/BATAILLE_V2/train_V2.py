import gymnasium as gym
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback

# Environnement pour le jeu de cartes mis à jour
class CardGameEnv(gym.Env):
    """
    Environnement du jeu de cartes :
      - L'agent dispose des 52 cartes (actions de 0 à 51), qui contiennent une information de rang et de couleur.
      - Une carte aléatoire (avec rang de 1 à 13 et couleur 0 à 3) est tirée pour la table.
      - Pour battre la carte sur la table, l'agent doit jouer une carte de la même couleur ET de rang supérieur.
      - Le reward est +1 si l'agent gagne, -1 s'il perd, 0 en cas d'égalité.
      
    La méthode step() retourne 5 valeurs pour assurer la compatibilité avec Gymnasium.
    """
    def __init__(self, episode_length=100):
        super(CardGameEnv, self).__init__()
        self.episode_length = episode_length
        self.current_step = 0
        # L'observation est maintenant un vecteur [rang, couleur] : rang entre 1 et 13, couleur entre 0 et 3
        self.observation_space = gym.spaces.Box(low=np.array([1, 0]), high=np.array([13, 3]), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(52)
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # Tirage aléatoire d'une carte pour la table : rang entre 1 et 13, couleur entre 0 et 3
        table_rank = np.random.randint(1, 14)
        table_suit = np.random.randint(0, 4)
        self.table_card = (table_rank, table_suit)
        return np.array(self.table_card, dtype=np.int32), {}
    
    def step(self, action):
        # Décodage de l'action : 
        # - Les actions 0 à 3 correspondent au rang 1, 4 à 7 au rang 2, etc.
        played_rank = (action // 4) + 1
        played_suit = action % 4
        
        table_rank, table_suit = self.table_card
        
        # Pour battre la carte sur la table, la couleur doit être identique ET le rang supérieur
        if played_suit == table_suit:
            if played_rank > table_rank:
                reward = 1
            elif played_rank == table_rank:
                reward = 0
            else:
                reward = -1
        else:
            reward = -1

        info = {"table_card": self.table_card, "played_card": (played_rank, played_suit)}
        self.current_step += 1
        
        terminated = False  # Fin naturelle d'un épisode
        truncated = self.current_step >= self.episode_length  # Limite d'étapes atteinte
        
        # Génération d'une nouvelle carte pour la table pour le prochain pas
        table_rank = np.random.randint(1, 14)
        table_suit = np.random.randint(0, 4)
        self.table_card = (table_rank, table_suit)
        obs = np.array(self.table_card, dtype=np.int32)
        
        return obs, reward, terminated, truncated, info

# Callback pour enregistrer les statistiques d'épisodes dans un fichier JSONL
class LoggingCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.log_file_path = log_file_path
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                with open(self.log_file_path, "a") as f:
                    f.write(json.dumps(info["episode"]) + "\n")
        return True

# Callback pour afficher une barre de progression avec tqdm
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TqdmCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.tqdm_bar = None

    def _on_training_start(self):
        self.tqdm_bar = tqdm(total=self.total_timesteps)

    def _on_step(self) -> bool:
        self.tqdm_bar.update(1)
        return True

    def _on_training_end(self):
        self.tqdm_bar.close()

# Fonction pour générer et sauvegarder les graphiques d'évolution de l'entraînement
def plot_logs(log_file, plot_dir):
    rewards = []
    lengths = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                data = json.loads(line)
                rewards.append(data.get("r", 0))
                lengths.append(data.get("l", 0))
    else:
        print("Fichier de log non trouvé.")
        return

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Graphique des rewards par épisode
    plt.figure(figsize=(10,5))
    plt.plot(rewards, label="Reward par épisode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Évolution des Rewards")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "episode_rewards.png"))
    plt.close()

    # Graphique de la longueur des épisodes
    plt.figure(figsize=(10,5))
    plt.plot(lengths, label="Longueur d'épisode")
    plt.xlabel("Episode")
    plt.ylabel("Nombre d'étapes")
    plt.title("Évolution de la Longueur des Épisodes")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "episode_lengths.png"))
    plt.close()

def main():
    # Création de l'environnement d'entraînement avec le wrapper pour enregistrer les statistiques
    env = CardGameEnv()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Environnement d'évaluation pour EvalCallback
    eval_env = CardGameEnv()
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    
    # Création du modèle PPO avec la policy MLP par défaut
    model = PPO("MlpPolicy", env, verbose=0)
    
    total_timesteps = 100000  # Nombre total de pas d'entraînement

    # Préparation des callbacks
    log_callback = LoggingCallback("training_logs.jsonl")
    tqdm_callback = TqdmCallback(total_timesteps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        eval_freq=5000,
        n_eval_episodes=10,
        verbose=1
    )
    callback = CallbackList([log_callback, tqdm_callback, eval_callback])
    
    # Lancement de l'entraînement
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Sauvegarde du modèle final (le meilleur est sauvegardé automatiquement par EvalCallback)
    model.save("final_model")

    # Génération des graphiques d'évolution de l'entraînement dans le dossier "plots"
    plot_logs("training_logs.jsonl", "plots")

if __name__ == "__main__":
    main()
