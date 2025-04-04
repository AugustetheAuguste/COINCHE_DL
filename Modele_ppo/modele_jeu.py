import gym
import numpy as np
from gym import spaces

#changer la class env et la faire fonctionner avec main_ia_jeu.py 
#dans cette classe les trois fonction doivent etre implementer 

# Fonction d'encodage des cartes
def encode_cartes(cartes):
    cartes_vect = np.zeros(32)  # Il y a 32 cartes en Coinche
    for carte in cartes:
        cartes_vect[carte] = 1
    return cartes_vect

def encode_obs(obs):
    """Convertit l'observation en vecteur utilisable par PPO"""
    # Cartes en main (One-Hot de taille 32)
    cartes_main_vect = encode_cartes(obs["cartes_main"])

    # Encodage de l’atout (4 valeurs possibles)
    atout_vect = np.zeros(4)
    atout_vect[obs["atout"]] = 1

    # Encodage du contrat (normalisé entre 0 et 1)
    contrat_vect = np.array([obs["contrat"] / 160])

    # Encodage des plis passés (One-Hot)
    plis_vect = encode_cartes([carte for pli in obs["plis"] for carte in pli])

    # Encodage des cartes posées dans le pli en cours
    cartes_posees_vect = encode_cartes(obs["cartes_posees"])

    # Encodage du tour (normalisé entre 0 et 1)
    tour_vect = np.array([obs["tour"] / 3])

    # Concaténation de tous les vecteurs
    observation_vect = np.concatenate([
        cartes_main_vect,
        atout_vect,
        contrat_vect,
        plis_vect,
        cartes_posees_vect,
        tour_vect
    ])

    return observation_vect

class CoincheJeuEnv(gym.Env):
    def __init__(self):
        super(CoincheJeuEnv, self).__init__()

        # Définition de l'espace d'observation (vecteur de taille 165)
        self.observation_space = spaces.Box(low=0, high=1, shape=(165,), dtype=np.float32)

        # Définition de l'espace d'action (jouer une carte parmi celles disponibles)
        self.action_space = spaces.Discrete(8)  # Max 8 cartes en main

    def reset(self):
        """ Initialise une nouvelle partie """
        self.state = {
            "cartes_main": [2, 15, 18, 30],  # Exemple de cartes en main
            "atout": 2,
            "contrat": 80,
            "coinche": 1,
            "plis": [[12, 25, 38, 50]],  # Plis déjà joués
            "cartes_posees": [],  # Cartes posées dans le pli en cours
            "tour": 2
        }
        return encode_obs(self.state)

    def step(self, action):
        """ Joue une carte et met à jour l'état du jeu """
        # Supposons que l'action correspond à une carte en main
        carte_jouee = self.state["cartes_main"][action]
        self.state["cartes_posees"].append(carte_jouee)

        # Supposons que le pli est terminé après 4 actions
        done = len(self.state["cartes_posees"]) == 4

        # Calcul de la récompense (ex: +1 si on gagne le pli)
        reward = 0
        if done:
            # Supposons que si on joue la carte la plus forte, on gagne le pli
            if max(self.state["cartes_posees"]) == carte_jouee:
                reward = 1

            # Ajouter le pli aux plis joués et réinitialiser
            self.state["plis"].append(self.state["cartes_posees"])
            self.state["cartes_posees"] = []

        # Retourner le nouvel état
        return encode_obs(self.state), reward, done, {}

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Créer l'environnement
env = CoincheJeuEnv()

# Entraînement du modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entraînement sur un certain nombre de timesteps
model.learn(total_timesteps=500000)

# Sauvegarder le modèle
model.save("ppo_coinche_jeu")

# Sauvegarder l'environnement
env.save("coinche_env")

# Charger le modèle préalablement sauvegardé
model = PPO.load("ppo_coinche_jeu")

# Réinitialiser l'environnement
obs = env.reset()

done = False
total_reward = 0

while not done:
    # Choisir l'action à partir de l'état actuel
    action, _ = model.predict(obs)
    
    # Effectuer l'action et obtenir le nouvel état et la récompense
    obs, reward, done, info = env.step(action)
    
    # Ajouter la récompense à l'accumulation
    total_reward += reward
    
    # Afficher l'état ou la progression
    env.render()

print(f"Récompense totale de la partie : {total_reward}")
