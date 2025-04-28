import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Configuration du logging en UTF-8
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    encoding='utf-8')
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# -----------------------------------------------------------
# Paramètres globaux du jeu
# -----------------------------------------------------------
NB_CARTES = 52           # Deck de 52 cartes uniques (rangs 2 à 14, 14 = As)
NB_CARTES_AGENT = 26     # Chaque joueur reçoit 26 cartes
NB_TOURS = 26            # Nombre de tours (une carte par tour)
SUITS = ["coeur", "trèfle", "carreau", "pique"]

# -----------------------------------------------------------
# Fonction d'encodage d'une carte
# -----------------------------------------------------------
def encode_card(card):
    """
    Encode une carte sous la forme d'un vecteur de dimension 5 :
      - La première valeur représente le rang normalisé (de 0 à 1) : 2 → 0.0, As (14) → 1.0.
      - Les 4 valeurs suivantes correspondent à un encodage one-hot de la couleur
        dans l'ordre : coeur, trèfle, carreau, pique.
    Si la carte est None, renvoie un vecteur de zéros.
    """
    if card is None:
        return np.zeros(5, dtype=np.float32)
    rank, suit = card
    norm_rank = (rank - 2) / 12.0
    one_hot = np.zeros(4, dtype=np.float32)
    try:
        suit_index = SUITS.index(suit)
    except ValueError:
        suit_index = -1
    if suit_index >= 0:
        one_hot[suit_index] = 1.0
    return np.concatenate(([norm_rank], one_hot))

# -----------------------------------------------------------
# Environnement : Jeu de la Bataille (adapté)
# -----------------------------------------------------------
class BatailleEnv:
    def __init__(self):
        self.nb_tours = NB_TOURS
        self.reset()
    
    def reset(self):
        logging.debug("Réinitialisation de l'environnement.")
        # Création d'un deck de 52 cartes (rangs 2 à 14)
        self.deck = [(rank, suit) for suit in SUITS for rank in range(2, 15)]
        random.shuffle(self.deck)
        # Distribution : première moitié pour l'adversaire, deuxième moitié pour l'agent
        self.adversaire_main = self.deck[:NB_CARTES_AGENT]
        self.agent_main = self.deck[NB_CARTES_AGENT:]
        self.tour = 0
        self.nouvelle_manche()
        return self.get_observation_vector()
    
    def nouvelle_manche(self):
        """L'adversaire joue une carte aléatoirement depuis sa main."""
        if len(self.adversaire_main) > 0:
            idx = random.randrange(len(self.adversaire_main))
            self.current_adversaire_carte = self.adversaire_main.pop(idx)
            logging.debug("Tour %d : L'adversaire joue %s", self.tour, self.current_adversaire_carte)
        else:
            self.current_adversaire_carte = None

    def get_observation_dict(self):
        """
        Observation sous forme de dictionnaire (pour les agents rule-based et aléatoire) :
          - 'agent_main' : la main de l'agent (liste de cartes ou None)
          - 'adversaire_card' : la carte jouée par l'adversaire ce tour
        """
        return {'agent_main': self.agent_main, 'adversaire_card': self.current_adversaire_carte}

    def get_observation_vector(self):
        """
        Observation encodée sur 135 dimensions (pour le modèle PPO) :
          - 26 cartes * 5 dimensions (agent_main) + 5 dimensions (carte de l'adversaire)
        """
        hand_obs = [encode_card(card) for card in self.agent_main]
        # Compléter si nécessaire
        while len(hand_obs) < NB_CARTES_AGENT:
            hand_obs.append(np.zeros(5, dtype=np.float32))
        hand_obs = np.concatenate(hand_obs)  # 26*5 = 130
        adv_obs = encode_card(self.current_adversaire_carte)  # 5 dimensions
        obs = np.concatenate([hand_obs, adv_obs])
        return obs.astype(np.float32)
    
    def get_valid_action_mask(self):
        """
        Retourne un masque de taille 26 indiquant les positions jouables dans la main de l'agent.
        - D'abord, seules les positions non vides sont valides.
        - Ensuite, si l'agent possède au moins une carte de la même couleur que celle de l'adversaire,
          seules ces positions seront marquées comme valides.
        """
        mask = np.zeros(NB_CARTES_AGENT, dtype=np.float32)
        for i in range(len(self.agent_main)):
            if self.agent_main[i] is not None:
                mask[i] = 1.0
        adv_suit = self.current_adversaire_carte[1] if self.current_adversaire_carte is not None else None
        possede_couleur = any(card is not None and card[1] == adv_suit for card in self.agent_main)
        if possede_couleur:
            for i in range(len(self.agent_main)):
                if self.agent_main[i] is not None and self.agent_main[i][1] != adv_suit:
                    mask[i] = 0.0
        logging.debug("Masque d'action valide : %s", mask)
        return mask
    
    def step(self, action):
        """
        L'agent joue la carte à l'index donné dans sa main.
        Règles :
          - Si la carte jouée est de la même couleur que celle de l'adversaire :
              * Si son rang est supérieur → reward = +1
              * Sinon → reward = -1
          - Sinon → reward = -1.
        La carte jouée est retirée de la main.
        """
        logging.debug("Action choisie par l'agent : %d", action)
        if action < 0 or action >= NB_CARTES_AGENT or self.agent_main[action] is None:
            reward = -1.0
            logging.debug("Action invalide : index hors limite ou carte absente.")
        else:
            carte_agent = self.agent_main[action]
            adv_carte = self.current_adversaire_carte
            logging.debug("Carte de l'agent : %s, Carte de l'adversaire : %s", carte_agent, adv_carte)
            if carte_agent[1] == adv_carte[1]:
                reward = 1.0 if carte_agent[0] > adv_carte[0] else -1.0
            else:
                reward = -1.0
            self.agent_main[action] = None
        
        self.tour += 1
        done = (self.tour >= self.nb_tours)
        if not done:
            self.nouvelle_manche()
        return self.get_observation_vector() if hasattr(self, 'using_vector') and self.using_vector else self.get_observation_dict(), reward, done, {}
    
    def render(self):
        logging.info("Tour : %d", self.tour)
        logging.info("Main de l'agent : %s", self.agent_main)
        logging.info("Carte de l'adversaire : %s", self.current_adversaire_carte)

# -----------------------------------------------------------
# Agent Rule-Based (stratégie optimale)
# -----------------------------------------------------------
class RuleBasedAgent:
    def select_action(self, observation):
        """
        L'agent reçoit l'observation sous forme de dictionnaire et retourne l'index de la carte à jouer.
        """
        hand = observation['agent_main']
        adv_card = observation['adversaire_card']
        adv_rank, adv_suit = adv_card
        same_suit = [(i, card) for i, card in enumerate(hand) if card is not None and card[1] == adv_suit]
        if same_suit:
            winning = [(i, card) for i, card in same_suit if card[0] > adv_rank]
            if winning:
                chosen = min(winning, key=lambda x: x[1][0])
                return chosen[0]
            else:
                chosen = min(same_suit, key=lambda x: x[1][0])
                return chosen[0]
        else:
            suit_counts = {}
            for card in hand:
                if card is not None:
                    suit_counts[card[1]] = suit_counts.get(card[1], 0) + 1
            if not suit_counts:
                return -1
            best_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            candidates = [(i, card) for i, card in enumerate(hand) if card is not None and card[1] == best_suit]
            chosen = min(candidates, key=lambda x: x[1][0])
            return chosen[0]

# -----------------------------------------------------------
# Agent Aléatoire
# -----------------------------------------------------------
class RandomAgent:
    def select_action(self, observation):
        """
        Sélectionne aléatoirement une carte valide dans la main de l'agent (observation dictionnaire).
        """
        hand = observation['agent_main']
        valid_indices = [i for i, card in enumerate(hand) if card is not None]
        return random.choice(valid_indices) if valid_indices else -1

# -----------------------------------------------------------
# Réseau Actor-Critic (pour le modèle PPO)
# -----------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

# -----------------------------------------------------------
# Agent basé sur votre modèle pré-entraîné
# -----------------------------------------------------------
class ModelAgent:
    def __init__(self, obs_dim=135, action_dim=NB_CARTES_AGENT, hidden_dim=128, model_path="best_model.pt"):
        self.model = ActorCriticNetwork(obs_dim, hidden_dim, action_dim)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def select_action(self, obs, valid_mask):
        """
        Pour une observation encodée et un masque d'action, retourne l'action choisie,
        la log-probabilité associée et la valeur estimée.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        logits, value = self.model(obs_tensor)
        logits = logits.squeeze(0)
        # Masquage : les positions invalides reçoivent une très faible valeur
        mask_tensor = torch.from_numpy(valid_mask).float()
        masked_logits = logits + (1 - mask_tensor) * (-1e8)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob, value.item()

# -----------------------------------------------------------
# Simulation de parties pour un agent donné
# -----------------------------------------------------------
def simulate_games(agent, num_games=10000):
    rewards = []
    win_rates = []
    for game in range(num_games):
        env = BatailleEnv()
        # Pour le modèle, on souhaite utiliser l'observation vectorielle
        if isinstance(agent, ModelAgent):
            env.using_vector = True
            obs = env.get_observation_vector()
        else:
            env.using_vector = False
            obs = env.get_observation_dict()
        total_reward = 0
        wins = 0
        for _ in range(NB_TOURS):
            if isinstance(agent, ModelAgent):
                valid_mask = env.get_valid_action_mask()
                action, _, _ = agent.select_action(obs, valid_mask)
                obs, reward, done, _ = env.step(action)
            else:
                action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                # Pour les agents non-modèle, reconvertir l'observation vectorielle en dict
                # en utilisant les attributs internes de l'env.
                obs = env.get_observation_dict()
            total_reward += reward
            if reward > 0:
                wins += 1
            if done:
                break
        rewards.append(total_reward)
        win_rates.append(wins / NB_TOURS)
    return rewards, win_rates

# -----------------------------------------------------------
# Boucle de test et affichage des résultats
# -----------------------------------------------------------
def main():
    num_games = 1000

    # Agents à tester
    agent_rule = RuleBasedAgent()
    agent_random = RandomAgent()
    # Pour le modèle, on part du principe que best_model.pt est dans le même dossier
    agent_model = ModelAgent(model_path="BATAILLE_V4\best_model.pt")

    logging.info("Simulation avec l'agent rule-based.")
    rewards_rule, win_rates_rule = simulate_games(agent_rule, num_games)
    logging.info("Simulation avec l'agent aléatoire.")
    rewards_random, win_rates_random = simulate_games(agent_random, num_games)
    logging.info("Simulation avec le modèle pré-entraîné.")
    rewards_model, win_rates_model = simulate_games(agent_model, num_games)
    
    # Graphique comparatif des rewards
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_rule, alpha=0.5, label="Rule-Based")
    plt.plot(np.convolve(rewards_rule, np.ones(100)/100, mode='valid'), color='red', label="Moyenne glissante (Rule-Based)")
    plt.plot(rewards_random, alpha=0.5, label="Aléatoire")
    plt.plot(np.convolve(rewards_random, np.ones(100)/100, mode='valid'), color='blue', label="Moyenne glissante (Aléatoire)")
    plt.plot(rewards_model, alpha=0.5, label="Modèle")
    plt.plot(np.convolve(rewards_model, np.ones(100)/100, mode='valid'), color='green', label="Moyenne glissante (Modèle)")
    plt.title("Reward total par partie")
    plt.xlabel("Partie")
    plt.ylabel("Reward total")
    plt.legend()
    
    # Histogramme des rewards
    plt.subplot(1, 2, 2)
    plt.hist(rewards_rule, bins=30, alpha=0.7, label="Rule-Based", edgecolor='black')
    plt.hist(rewards_random, bins=30, alpha=0.7, label="Aléatoire", edgecolor='black')
    plt.hist(rewards_model, bins=30, alpha=0.7, label="Modèle", edgecolor='black')
    plt.title("Distribution des rewards sur {} parties".format(num_games))
    plt.xlabel("Reward total")
    plt.ylabel("Nombre de parties")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Graphique comparatif des win rates
    plt.figure(figsize=(7, 5))
    plt.plot(win_rates_rule, alpha=0.5, label="Rule-Based")
    plt.plot(np.convolve(win_rates_rule, np.ones(100)/100, mode='valid'), color='red', label="Moyenne glissante (Rule-Based)")
    plt.plot(win_rates_random, alpha=0.5, label="Aléatoire")
    plt.plot(np.convolve(win_rates_random, np.ones(100)/100, mode='valid'), color='blue', label="Moyenne glissante (Aléatoire)")
    plt.plot(win_rates_model, alpha=0.5, label="Modèle")
    plt.plot(np.convolve(win_rates_model, np.ones(100)/100, mode='valid'), color='green', label="Moyenne glissante (Modèle)")
    plt.title("Win rate par partie")
    plt.xlabel("Partie")
    plt.ylabel("Win rate (fraction de tricks gagnés)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
