import tkinter as tk
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Fonction pour convertir une carte (tuple: (rang, couleur)) en chaîne lisible
def card_to_str(card):
    rank, suit = card
    rank_names = {
        1: "As",
        11: "Valet",
        12: "Dame",
        13: "Roi"
    }
    # Si le rang n'est pas dans rank_names, on garde la valeur numérique
    rank_str = rank_names.get(rank, str(rank))
    suits = {
        0: "Trèfle",
        1: "Carreau",
        2: "Coeur",
        3: "Pique"
    }
    suit_str = suits.get(suit, str(suit))
    return f"{rank_str} de {suit_str}"

# Wrapper personnalisé pour aplatir l'observation (dictionnaire) en un vecteur plat
class CustomFlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # "table_card" contient 2 valeurs et "hand" 52 valeurs → vecteur de 54 dimensions
        flat_shape = 2 + 52
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_shape,), dtype=np.int32)

    def observation(self, observation):
        table = observation["table_card"].flatten()
        hand = observation["hand"].flatten()
        return np.concatenate([table, hand])

# Environnement du jeu de cartes avec main complète
class CardGameEnv(gym.Env):
    """
    Environnement du jeu de cartes :
      - Observation : dictionnaire avec "table_card" ([rang, couleur]) et "hand" (vecteur binaire de taille 52).
      - Action : entier de 0 à 51 correspondant à une carte.
      - Reward : +1 si l'agent gagne (joue une carte de même couleur et de rang supérieur),
                 0 en cas d'égalité et -1 sinon.
    """
    def __init__(self, episode_length=100):
        super(CardGameEnv, self).__init__()
        self.episode_length = episode_length
        self.current_step = 0
        self.observation_space = gym.spaces.Dict({
            "table_card": gym.spaces.Box(low=np.array([1, 0]), high=np.array([13, 3]), dtype=np.int32),
            "hand": gym.spaces.MultiBinary(52)
        })
        self.action_space = gym.spaces.Discrete(52)
        self.reset_hand()
    
    def reset_hand(self):
        # Toutes les 52 cartes sont disponibles (1 = disponible)
        self.hand = np.ones(52, dtype=np.int32)
    
    # Méthode reset conforme à Gymnasium
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self.reset_hand()
        table_rank = np.random.randint(1, 14)
        table_suit = np.random.randint(0, 4)
        self.table_card = (table_rank, table_suit)
        obs = {
            "table_card": np.array(self.table_card, dtype=np.int32),
            "hand": self.hand.copy()
        }
        return obs, {}
    
    def step(self, action):
        # Décodage de l'action
        played_rank = (action // 4) + 1
        played_suit = action % 4
        table_rank, table_suit = self.table_card

        # Règle de victoire : même couleur ET rang supérieur
        if played_suit == table_suit:
            if played_rank > table_rank:
                reward = 1
            elif played_rank == table_rank:
                reward = 0
            else:
                reward = -1
        else:
            reward = -1

        # On stocke la carte jouée et la carte sur la table sous forme de tuple
        info = {
            "table_card": self.table_card,
            "played_card": (played_rank, played_suit)
        }
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.episode_length

        # Nouvelle carte sur la table pour le prochain tour
        table_rank = np.random.randint(1, 14)
        table_suit = np.random.randint(0, 4)
        self.table_card = (table_rank, table_suit)

        obs = {
            "table_card": np.array(self.table_card, dtype=np.int32),
            "hand": self.hand.copy()
        }
        return obs, reward, terminated, truncated, info

def main():
    # Création de l'environnement et application du wrapper personnalisé
    env = CardGameEnv()
    env = CustomFlattenObservation(env)
    obs, _ = env.reset()

    # Chargement du modèle entraîné
    model_path = "./best_model/best_model.zip"
    model = PPO.load(model_path)

    # Création de l'interface graphique avec Tkinter
    root = tk.Tk()
    root.title("Visualisation de l'Agent - Jeu de Cartes")
    root.geometry("400x300")

    # Labels pour afficher les informations
    table_label = tk.Label(root, text="Carte sur la table : ?", font=("Helvetica", 20))
    table_label.pack(pady=10)
    agent_label = tk.Label(root, text="Carte jouée par l'agent : ?", font=("Helvetica", 20))
    agent_label.pack(pady=10)
    reward_label = tk.Label(root, text="Reward : ?", font=("Helvetica", 20))
    reward_label.pack(pady=10)

    # Fonction pour passer au tour suivant
    def next_round():
        nonlocal obs
        # L'agent prédit l'action à partir de l'observation (vecteur de dimension 54)
        action, _ = model.predict(obs, deterministic=True)
        new_obs, reward, terminated, truncated, info = env.step(action)
        table_label.config(text=f"Carte sur la table : {card_to_str(info['table_card'])}")
        agent_label.config(text=f"Carte jouée par l'agent : {card_to_str(info['played_card'])}")
        reward_label.config(text=f"Reward : {reward}")
        obs = new_obs
        if terminated or truncated:
            obs, _ = env.reset()

    next_button = tk.Button(root, text="Next Round", command=next_round, font=("Helvetica", 16))
    next_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
