import gym
import time
import numpy as np
from stable_baselines3 import PPO
from modele_jeu_v2 import CoincheEnv

def train_agent():
    """Entraîne l'IA à jouer à la Coinche."""
    env = CoincheEnv()
    model = PPO("MlpPolicy", env, verbose=1)

    print("Démarrage de l'entraînement...")
    model.learn(total_timesteps=10000, log_interval=None)  # Ajuster selon la puissance de calcul
    model.save("coinche_ia_v2")

    print("Entraînement terminé !")

def test_agent():
    """Teste l'IA après entraînement."""
    env = CoincheEnv()
    model = PPO.load("coinche_ia")

    obs = env.reset()
    done = False
    for player in env.game.players:
        print([str(card) for card in player.get_card()])
    print(env.game.table.get_current_bid().get_trump_suit())
    print(env.game.table.get_current_bid().get_player())
    tour = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if reward != -10:
            tour+=1
        if tour == 4:
            tour = 0
            print("carte joué : ",env.current_player,[str(card) for card in env.game.table.get_all_cards()[-4:]])
            for player in env.game.players:
                print([str(card) for card in player.get_card()])
        env.render()
        # time.sleep(1)

def choose_valid_action(model, obs, legal_actions):
    action, _ = model.predict(obs)  # Prédiction brute

    # Vérifier si l'action est valide
    if action not in legal_actions:
        action = np.random.choice(legal_actions)  # Sélectionner une action valide au hasard
    
    return action


if __name__ == "__main__":
    train_agent()
    test_agent()
