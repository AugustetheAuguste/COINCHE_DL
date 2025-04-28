import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Configuration du logging en UTF-8
# -----------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    encoding='utf-8')

# Réduire les logs de debug de Matplotlib concernant les fonts
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# -----------------------------------------------------------
# Paramètres globaux du jeu
# -----------------------------------------------------------
NB_CARTES = 52           # Deck de 52 cartes uniques
NB_CARTES_AGENT = 26     # Chaque joueur reçoit 26 cartes (la moitié du deck)
NB_TOURS = 26            # Nombre de tours de jeu = 26
SUITS = ["coeur", "trèfle", "carreau", "pique"]

# -----------------------------------------------------------
# Fonction d'encodage d'une carte
# -----------------------------------------------------------
def encode_card(card):
    """
    Encode une carte sous la forme d'un vecteur de dimension 5 :
      - La première valeur représente le rang normalisé (de 0 à 1), où 2 -> 0.0 et As (14) -> 1.0.
      - Les 4 valeurs suivantes sont un encodage one-hot de la couleur, dans l'ordre : coeur, trèfle, carreau, pique.
    Si la carte est None (pas de carte), renvoie un vecteur de zéros.
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
# Environnement : Le jeu de la bataille
# -----------------------------------------------------------
class BatailleEnv:
    def __init__(self):
        self.nb_tours = NB_TOURS
        self.reset()
    
    def reset(self):
        logging.debug("Réinitialisation de l'environnement.")
        # Création d'un deck de 52 cartes uniques
        self.deck = [(rank, suit) for suit in SUITS for rank in range(2, 15)]
        random.shuffle(self.deck)
        logging.debug("Deck mélangé : %s", self.deck)
        # Distribution : la première moitié pour l'adversaire, la deuxième moitié pour l'agent
        self.adversaire_main = self.deck[:NB_CARTES_AGENT]
        self.agent_main = self.deck[NB_CARTES_AGENT:]
        logging.debug("Main de l'adversaire : %s", self.adversaire_main)
        logging.debug("Main de l'agent : %s", self.agent_main)
        self.tour = 0
        # Lancer la première manche : l'adversaire joue d'abord
        self.nouvelle_manche()
        return self.get_observation()
    
    def nouvelle_manche(self):
        """
        L'adversaire joue une carte aléatoirement depuis sa main.
        """
        if len(self.adversaire_main) > 0:
            idx = random.randrange(len(self.adversaire_main))
            self.current_adversaire_carte = self.adversaire_main.pop(idx)
            logging.debug("Tour %d : L'adversaire joue %s", self.tour, self.current_adversaire_carte)
        else:
            self.current_adversaire_carte = None
    
    def get_observation(self):
        """
        L'observation se compose de :
          - La main de l'agent sous forme de 26 vecteurs (5 dimensions chacun)
          - La carte actuelle de l'adversaire (vecteur de 5 dimensions)
        Le vecteur final a donc une dimension de 26*5 + 5 = 135.
        """
        hand_obs = []
        for card in self.agent_main:
            hand_obs.append(encode_card(card))
        # Compléter avec des zéros si l'agent n'a plus de carte (bien que ce cas ne devrait pas arriver ici)
        while len(hand_obs) < NB_CARTES_AGENT:
            hand_obs.append(np.zeros(5, dtype=np.float32))
        hand_obs = np.concatenate(hand_obs)  # Dimension : 26*5 = 130
        adv_obs = encode_card(self.current_adversaire_carte)  # Dimension : 5
        obs = np.concatenate([hand_obs, adv_obs])  # Dimension totale : 135
        return obs.astype(np.float32)
    
    def get_valid_action_mask(self):
        """
        Retourne un masque d'actions de taille 26 (pour la main de l'agent) indiquant les positions jouables.
        - D'abord, seules les positions contenant une carte sont marquées comme valides.
        - Ensuite, si l'agent possède au moins une carte de la même couleur que celle jouée par l'adversaire,
          seules ces positions seront valides.
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
        L'agent joue une carte en indiquant l'index dans sa main.
        Règles du jeu :
          - Si la carte jouée est de la même couleur que celle de l'adversaire :
              * Si le rang de la carte de l'agent est supérieur à celui de l'adversaire → reward = +1
              * Sinon → reward = -1
          - Si la carte jouée n'est pas de la même couleur, l'agent perd (reward = -1)
        La carte jouée est retirée de la main de l'agent.
        """
        logging.debug("Action choisie par l'agent : %d", action)
        done = False
        
        if action < 0 or action >= NB_CARTES_AGENT or self.agent_main[action] is None:
            reward = -1.0
            logging.debug("Action invalide : index hors limite ou carte absente.")
        else:
            carte_agent = self.agent_main[action]
            adv_carte = self.current_adversaire_carte
            logging.debug("Carte de l'agent : %s, Carte de l'adversaire : %s", carte_agent, adv_carte)
            if carte_agent[1] == adv_carte[1]:
                if carte_agent[0] > adv_carte[0]:
                    reward = 1.0
                    logging.debug("Gagné : Carte de l'agent plus haute.")
                else:
                    reward = -1.0
                    logging.debug("Perdu : Carte de l'agent moins haute ou de rang égal.")
            else:
                reward = -1.0
                logging.debug("Perdu : L'agent n'a pas suivi la couleur.")
            # Retirer la carte jouée de la main de l'agent
            self.agent_main[action] = None
        
        self.tour += 1
        if self.tour >= self.nb_tours:
            done = True
            logging.debug("Fin de la partie après %d tours.", self.tour)
        else:
            self.nouvelle_manche()  # L'adversaire joue la prochaine carte
        obs = self.get_observation()
        return obs, reward, done, {}
    
    def render(self):
        logging.info("Tour : %d", self.tour)
        logging.info("Main de l'agent : %s", self.agent_main)
        logging.info("Carte actuelle de l'adversaire : %s", self.current_adversaire_carte)

# -----------------------------------------------------------
# Réseau de politique (PolicyNetwork)
# -----------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

# -----------------------------------------------------------
# Agent PPO avec masquage d'action
# -----------------------------------------------------------
class PPOAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim=128, lr=1e-3):
        self.policy_net = PolicyNetwork(obs_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.eps_clip = 0.2
        self.gamma = 0.99
    
    def select_action(self, obs, valid_mask):
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # Dimension : (1, obs_dim)
        logits = self.policy_net(obs_tensor).squeeze(0)           # Dimension : (action_dim,)
        logging.debug("Logits bruts : %s", logits.detach().numpy())
        
        mask_tensor = torch.from_numpy(valid_mask).float()
        # Pour les actions invalides, on affecte une très faible valeur
        masked_logits = logits + (1 - mask_tensor) * (-1e8)
        logging.debug("Logits masqués : %s", masked_logits.detach().numpy())
        
        distribution = torch.distributions.Categorical(logits=masked_logits)
        action = distribution.sample()
        action_log_prob = distribution.log_prob(action)
        logging.debug("Action sélectionnée : %d, log prob : %.4f", action.item(), action_log_prob.item())
        return action.item(), action_log_prob, distribution
    
    def compute_returns(self, rewards, dones, next_value=0):
        R = next_value
        returns = []
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns
    
    def update(self, trajectories):
        observations = torch.tensor(trajectories['observations'], dtype=torch.float32)
        actions = torch.tensor(trajectories['actions'], dtype=torch.long)
        old_log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float32)
        returns = torch.tensor(trajectories['returns'], dtype=torch.float32)
        
        advantages = returns - returns.mean()
        
        logits = self.policy_net(observations)
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        logging.debug("Loss de politique : %.4f", policy_loss.item())
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        return policy_loss.item()

# -----------------------------------------------------------
# Boucle d'entraînement principale
# -----------------------------------------------------------
def main():
    env = BatailleEnv()
    # Observation : 26 cartes * 5 dimensions + 5 dimensions (carte de l'adversaire) = 135 dimensions
    obs_dim = 135
    action_dim = NB_CARTES_AGENT  # 26 actions possibles (index dans la main)
    agent = PPOAgent(obs_dim, action_dim, hidden_dim=128, lr=1e-3)
    
    num_episodes = 10_000  # Nombre d'épisodes d'entraînement
    all_rewards = []
    all_losses = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        
        trajectoire = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }
        total_reward = 0
        
        logging.info("Début de l'épisode %d", ep)
        while not done and step < NB_TOURS:
            valid_mask = env.get_valid_action_mask()
            action, log_prob, _ = agent.select_action(obs, valid_mask)
            
            trajectoire['observations'].append(obs)
            trajectoire['actions'].append(action)
            trajectoire['log_probs'].append(log_prob.item())
            
            new_obs, reward, done, _ = env.step(action)
            trajectoire['rewards'].append(reward)
            trajectoire['dones'].append(done)
            total_reward += reward
            
            logging.info("Épisode %d - Tour %d : Action %d, Reward : %.2f, Reward total : %.2f, Done : %s",
                         ep, step, action, reward, total_reward, done)
            obs = new_obs
            step += 1
        
        returns = agent.compute_returns(trajectoire['rewards'], trajectoire['dones'])
        trajectoire['returns'] = returns
        
        loss = agent.update(trajectoire)
        all_rewards.append(total_reward)
        all_losses.append(loss)
        logging.info("Fin de l'épisode %d : Reward total = %.2f, Loss = %.4f", ep, total_reward, loss)
    
    # Tracer les courbes du reward total et de la loss par épisode
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(all_rewards)
    plt.title("Reward total par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Reward")
    
    plt.subplot(1,2,2)
    plt.plot(all_losses)
    plt.title("Loss par épisode")
    plt.xlabel("Épisode")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
