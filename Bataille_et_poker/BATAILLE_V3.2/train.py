import gymnasium as gym
import numpy as np
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from gymnasium.wrappers import FlattenObservation

# Environnement du jeu de cartes mis à jour avec pénalisation des actions illégales
class CardGameEnv(gym.Env):
    """
    Environnement du jeu de cartes avec main limitée et pénalisation des actions illégales :
      - L'agent dispose d'une main de 52 cartes (actions de 0 à 51) indiquée par un vecteur binaire.
        Une carte disponible vaut 1, une carte déjà jouée vaut 0.
      - L'observation contient :
          * "table_card": la carte sur la table [rang, couleur]
          * "hand": vecteur binaire de taille 52 indiquant les cartes disponibles
          * "action_mask": identique à "hand" pour informer des actions légales
      - Une carte aléatoire (rang entre 1 et 13 et couleur entre 0 et 3) est tirée pour la table.
      - Pour battre la carte sur la table, l'agent doit jouer une carte de la même couleur ET de rang supérieur.
      - Il y a égalité si la carte jouée est de la même hauteur et de la même couleur.
      - L'agent perd s'il joue une carte de la même couleur mais de rang inférieur.
      - Une action est illégale si l'agent joue une carte d'une couleur différente ou une carte déjà jouée.
      - En cas d'action illégale, le reward est de -10 et la main (ou la table) n'est pas modifiée (sauf pour retirer la carte dans le cas d'une couleur différente).
      - Lorsque toutes les cartes ont été jouées, la main est réinitialisée.
      - Un compteur interne suit le nombre de coups illégaux dans l'épisode.
    """
    def __init__(self, episode_length=100):
        super(CardGameEnv, self).__init__()
        self.episode_length = episode_length
        self.current_step = 0
        
        self.observation_space = gym.spaces.Dict({
            "table_card": gym.spaces.Box(low=np.array([1, 0]), high=np.array([13, 3]), dtype=np.int32),
            "hand": gym.spaces.MultiBinary(52),
            "action_mask": gym.spaces.MultiBinary(52)
        })
        
        self.action_space = gym.spaces.Discrete(52)
        self.reset_hand()
        self.illegal_moves_count = 0

    def reset_hand(self):
        self.hand = np.ones(52, dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.reset_hand()
        self.illegal_moves_count = 0
        table_rank = np.random.randint(1, 14)
        table_suit = np.random.randint(0, 4)
        self.table_card = (table_rank, table_suit)
        obs = {
            "table_card": np.array(self.table_card, dtype=np.int32),
            "hand": self.hand.copy(),
            "action_mask": self.hand.copy()
        }
        return obs, {}

    def step(self, action):
        illegal_move = False

        # Vérifier si la carte a déjà été jouée
        if self.hand[action] == 0:
            illegal_move = True
            reward = -10
            self.illegal_moves_count += 1
            info = {
                "error": "Carte déjà jouée",
                "table_card": self.table_card,
                "played_card": None,
                "hand_reset": False,
                "illegal_move": True
            }
            obs = {
                "table_card": np.array(self.table_card, dtype=np.int32),
                "hand": self.hand.copy(),
                "action_mask": self.hand.copy()
            }
        else:
            played_rank = (action // 4) + 1
            played_suit = action % 4
            table_rank, table_suit = self.table_card

            # Vérifier si la carte jouée est de la même couleur que la carte sur la table
            if played_suit != table_suit:
                illegal_move = True
                reward = -10
                self.illegal_moves_count += 1
                info = {
                    "error": "Carte de couleur différente",
                    "table_card": self.table_card,
                    "played_card": (played_rank, played_suit),
                    "hand_reset": False,
                    "illegal_move": True
                }
                # Retirer la carte même si le coup est illégal
                self.hand[action] = 0
                obs = {
                    "table_card": np.array(self.table_card, dtype=np.int32),
                    "hand": self.hand.copy(),
                    "action_mask": self.hand.copy()
                }
            else:
                # Action légale : la carte est de la bonne couleur
                self.hand[action] = 0
                if played_rank > table_rank:
                    reward = 1
                elif played_rank == table_rank:
                    reward = 0
                else:
                    reward = -1
                info = {
                    "table_card": self.table_card,
                    "played_card": (played_rank, played_suit),
                    "hand_reset": False,
                    "illegal_move": False
                }
                obs = {
                    "table_card": np.array(self.table_card, dtype=np.int32),
                    "hand": self.hand.copy(),
                    "action_mask": self.hand.copy()
                }
                
        # Incrémenter le compteur de pas
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length

        # Réinitialisation de la main si toutes les cartes ont été jouées
        if np.sum(self.hand) == 0:
            self.reset_hand()
            info["hand_reset"] = True

        # Si l'épisode se termine, ajouter le nombre de coups illégaux dans l'info
        if truncated:
            info["illegal_moves"] = self.illegal_moves_count
            self.illegal_moves_count = 0

        # Mise à jour de la carte sur la table uniquement en cas d'action légale
        if not illegal_move:
            table_rank = np.random.randint(1, 14)
            table_suit = np.random.randint(0, 4)
            self.table_card = (table_rank, table_suit)
            obs["table_card"] = np.array(self.table_card, dtype=np.int32)

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

# Fonction pour générer et sauvegarder les graphiques d'évolution de l'entraînement,
# incluant un graphe du nombre de coups illégaux par épisode
def plot_logs(log_file, plot_dir):
    rewards = []
    lengths = []
    illegal_moves = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            for line in f:
                data = json.loads(line)
                rewards.append(data.get("r", 0))
                lengths.append(data.get("l", 0))
                illegal_moves.append(data.get("illegal_moves", 0))
    else:
        print("Fichier de log non trouvé.")
        return

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Graphique des rewards par épisode
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Reward")
    plt.title("Évolution des Rewards")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "episode_rewards.png"))
    plt.close()

    # Graphique de la longueur des épisodes
    plt.figure(figsize=(10, 5))
    plt.plot(lengths, label="Longueur d'épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Nombre d'étapes")
    plt.title("Évolution de la Longueur des Épisodes")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "episode_lengths.png"))
    plt.close()

    # Graphique du nombre de coups illégaux par épisode
    plt.figure(figsize=(10, 5))
    plt.plot(illegal_moves, label="Coups illégaux par épisode", color="red")
    plt.xlabel("Épisode")
    plt.ylabel("Nombre de coups illégaux")
    plt.title("Évolution des Coups Illégaux par Épisode")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "illegal_moves.png"))
    plt.close()

def main():
    # Création de l'environnement et application des wrappers :
    # - FlattenObservation transforme l'observation (dict) en vecteur plat.
    # - RecordEpisodeStatistics enregistre les stats d'épisodes.
    env = CardGameEnv()
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    eval_env = CardGameEnv()
    eval_env = gym.wrappers.FlattenObservation(eval_env)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    
    # Utilisation du GPU via device="cuda" si disponible (ici device="cpu" pour la compatibilité)
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    
    total_timesteps = 100_000

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
    
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    model.save("final_model")
    plot_logs("training_logs.jsonl", "plots")

if __name__ == "__main__":
    main()
