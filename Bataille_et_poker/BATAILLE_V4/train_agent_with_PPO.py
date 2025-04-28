import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------
# Configuration du logging en UTF-8
# -----------------------------------------------------------
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    encoding='utf-8')
# Réduire les logs de debug de Matplotlib sur les fonts
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# -----------------------------------------------------------
# Paramètres globaux du jeu et PPO
# -----------------------------------------------------------
NB_CARTES = 52           # Deck de 52 cartes uniques
NB_CARTES_AGENT = 26     # Chaque joueur reçoit 26 cartes
NB_TOURS = 26            # Nombre de tours (une carte par tour)
SUITS = ["coeur", "trèfle", "carreau", "pique"]

# Hyperparamètres PPO
GAMMA = 0.99
CLIP_EPSILON = 0.2
BATCH_EPISODES = 2      # Nombre d'épisodes collectés avant mise à jour
NUM_EPOCHS = 10          # Nombre d'epochs d'optimisation sur le batch
MINIBATCH_SIZE = 2      # Taille du mini-batch lors de l'optimisation
LR = 3e-4                # Taux d'apprentissage
VF_COEF = 0.5            # Coefficient de la loss de valeur
ENTROPY_COEF = 0.01      # Coefficient bonus d'entropie

# Dossier de sauvegarde
SAVE_PATH = "BATAILLE_V4\best_model.pt"

# -----------------------------------------------------------
# Fonction d'encodage d'une carte
# -----------------------------------------------------------
def encode_card(card):
    """
    Encode une carte sous la forme d'un vecteur de dimension 5 :
      - La première valeur représente le rang normalisé (de 0 à 1), où 2 → 0.0 et As (14) → 1.0.
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
# Environnement : Jeu de la Bataille
# -----------------------------------------------------------
class BatailleEnv:
    def __init__(self):
        self.nb_tours = NB_TOURS
        self.reset()
    
    def reset(self):
        logging.debug("Réinitialisation de l'environnement.")
        # Création d'un deck de 52 cartes uniques (rangs 2 à 14, 14 = As)
        self.deck = [(rank, suit) for suit in SUITS for rank in range(2, 15)]
        random.shuffle(self.deck)
        logging.debug("Deck mélangé : %s", self.deck)
        # Distribution : première moitié pour l'adversaire, deuxième moitié pour l'agent
        self.adversaire_main = self.deck[:NB_CARTES_AGENT]
        self.agent_main = self.deck[NB_CARTES_AGENT:]
        logging.debug("Main de l'adversaire : %s", self.adversaire_main)
        logging.debug("Main de l'agent : %s", self.agent_main)
        self.tour = 0
        # L'adversaire joue en premier
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
          - La main de l'agent : 26 cartes (chacune encodée sur 5 dimensions) → 130 dimensions.
          - La carte actuelle de l'adversaire (5 dimensions).
        Total : 135 dimensions.
        """
        hand_obs = []
        for card in self.agent_main:
            hand_obs.append(encode_card(card))
        # Si l'agent a moins de 26 cartes, compléter avec des zéros
        while len(hand_obs) < NB_CARTES_AGENT:
            hand_obs.append(np.zeros(5, dtype=np.float32))
        hand_obs = np.concatenate(hand_obs)  # 26*5 = 130
        adv_obs = encode_card(self.current_adversaire_carte)  # 5 dimensions
        obs = np.concatenate([hand_obs, adv_obs])
        return obs.astype(np.float32)
    
    def get_valid_action_mask(self):
        """
        Retourne un masque (taille 26) indiquant les positions jouables dans la main de l'agent.
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
          - Si la carte jouée n'est pas de la même couleur (ce qui ne devrait arriver que si l'agent n'en a pas),
            reward = -1.
        La carte jouée est retirée de la main.
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
                    logging.debug("Perdu : Carte de l'agent moins haute ou égale.")
            else:
                reward = -1.0
                logging.debug("Perdu : L'agent n'a pas suivi la couleur.")
            # Retirer la carte jouée
            self.agent_main[action] = None
        
        self.tour += 1
        if self.tour >= self.nb_tours:
            done = True
            logging.debug("Fin de la partie après %d tours.", self.tour)
        else:
            self.nouvelle_manche()  # Prochaine manche, l'adversaire joue
        obs = self.get_observation()
        return obs, reward, done, {}
    
    def render(self):
        logging.info("Tour : %d", self.tour)
        logging.info("Main de l'agent : %s", self.agent_main)
        logging.info("Carte actuelle de l'adversaire : %s", self.current_adversaire_carte)

# -----------------------------------------------------------
# Réseau Actor-Critic (partagé)
# -----------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        # Tronc commun
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Tête pour la politique
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        # Tête pour la valeur
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)  # value: (batch,)

# -----------------------------------------------------------
# Agent PPO complet avec collecte multi-épisodes et mise à jour en mini-batch
# -----------------------------------------------------------
class PPOAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim=128, lr=LR):
        self.model = ActorCriticNetwork(obs_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_epsilon = CLIP_EPSILON
        self.gamma = GAMMA
        self.vf_coef = VF_COEF
        self.entropy_coef = ENTROPY_COEF
    
    def select_action(self, obs, valid_mask):
        """
        Pour une observation et un masque d'action, retourne l'action choisie, la log-probabilité associée et la valeur.
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # (1, obs_dim)
        logits, value = self.model(obs_tensor)
        logits = logits.squeeze(0)
        value = value.item()
        logging.debug("Logits bruts : %s", logits.detach().numpy())
        
        # Appliquer le masque d'action
        mask_tensor = torch.from_numpy(valid_mask).float()
        masked_logits = logits + (1 - mask_tensor) * (-1e8)
        logging.debug("Logits masqués : %s", masked_logits.detach().numpy())
        
        distribution = torch.distributions.Categorical(logits=masked_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        logging.debug("Action sélectionnée : %d, log prob : %.4f, Valeur estimée : %.4f", 
                      action.item(), log_prob.item(), value)
        return action.item(), log_prob, value
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        Calcule les retours G_t et les avantages A_t = G_t - V(s_t) pour une trajectoire.
        """
        returns = []
        G = 0
        # Calculer en rétro-propagation sur l'épisode
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values
        return returns, advantages
    
    def update(self, batch):
        """
        Mise à jour PPO sur un batch de transitions collectées sur plusieurs épisodes.
        """
        observations = torch.tensor(batch['observations'], dtype=torch.float32)
        actions = torch.tensor(batch['actions'], dtype=torch.long)
        old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32)
        returns = torch.tensor(batch['returns'], dtype=torch.float32)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32)
        
        total_samples = observations.shape[0]
        indices = np.arange(total_samples)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_idx = indices[start:end]
                mb_obs = observations[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                
                logits, values = self.model(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, mb_returns)
                
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
        
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        return avg_policy_loss, avg_value_loss, avg_entropy

    def collect_trajectories(self, env, batch_episodes=BATCH_EPISODES):
        """
        Collecte des transitions sur un certain nombre d'épisodes.
        Retourne un batch et le reward moyen sur ces épisodes.
        """
        batch = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'returns': [],
            'advantages': []
        }
        episode_rewards = []
        for ep in range(batch_episodes):
            obs = env.reset()
            done = False
            ep_obs = []
            ep_actions = []
            ep_log_probs = []
            ep_rewards_list = []
            ep_dones = []
            ep_values = []
            while not done:
                valid_mask = env.get_valid_action_mask()
                action, log_prob, value = self.select_action(obs, valid_mask)
                ep_obs.append(obs)
                ep_actions.append(action)
                ep_log_probs.append(log_prob.item())
                ep_values.append(value)
                obs, reward, done, _ = env.step(action)
                ep_rewards_list.append(reward)
                ep_dones.append(done)
            episode_rewards.append(sum(ep_rewards_list))
            returns, advantages = self.compute_returns_and_advantages(ep_rewards_list, ep_values, ep_dones)
            batch['observations'].extend(ep_obs)
            batch['actions'].extend(ep_actions)
            batch['log_probs'].extend(ep_log_probs)
            batch['rewards'].extend(ep_rewards_list)
            batch['dones'].extend(ep_dones)
            batch['values'].extend(ep_values)
            batch['returns'].extend(returns.numpy().tolist())
            batch['advantages'].extend(advantages.numpy().tolist())
            logging.info("Episode collecté: Reward total = %.2f", sum(ep_rewards_list))
        avg_reward = np.mean(episode_rewards)
        logging.info("Moyenne des rewards sur %d épisodes collectés: %.2f", batch_episodes, avg_reward)
        return batch, avg_reward

# -----------------------------------------------------------
# Boucle d'entraînement principale avec sauvegarde du meilleur modèle
# -----------------------------------------------------------
def main():
    env = BatailleEnv()
    # Observation : 26 cartes * 5 + 5 = 135 dimensions
    obs_dim = 135
    action_dim = NB_CARTES_AGENT  # 26 actions possibles
    agent = PPOAgent(obs_dim, action_dim, hidden_dim=128, lr=LR)
    
    num_updates = 1000  # Nombre d'itérations de mise à jour
    rewards_history = []
    policy_losses_history = []
    value_losses_history = []
    entropy_history = []
    
    best_reward = -float('inf')
    
    for update in range(num_updates):
        logging.info("=== Mise à jour %d ===", update)
        batch, avg_reward = agent.collect_trajectories(env, batch_episodes=BATCH_EPISODES)
        rewards_history.append(avg_reward)
        avg_policy_loss, avg_value_loss, avg_entropy = agent.update(batch)
        policy_losses_history.append(avg_policy_loss)
        value_losses_history.append(avg_value_loss)
        entropy_history.append(avg_entropy)
        logging.info("Update %d : Reward moyen = %.2f, Policy Loss = %.4f, Value Loss = %.4f, Entropy = %.4f",
                     update, avg_reward, avg_policy_loss, avg_value_loss, avg_entropy)
        
        # Sauvegarder le modèle si le reward moyen est le meilleur observé
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.model.state_dict(), SAVE_PATH)
            logging.info("Nouveau meilleur modèle sauvegardé avec un reward moyen de %.2f", best_reward)
    
    # Tracer les courbes de reward, de loss et d'entropie
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,3,1)
    plt.plot(rewards_history)
    plt.title("Reward moyen par mise à jour")
    plt.xlabel("Mise à jour")
    plt.ylabel("Reward moyen")
    
    plt.subplot(1,3,2)
    plt.plot(policy_losses_history, label="Policy Loss")
    plt.plot(value_losses_history, label="Value Loss")
    plt.title("Loss par mise à jour")
    plt.xlabel("Mise à jour")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(entropy_history)
    plt.title("Entropie moyenne par mise à jour")
    plt.xlabel("Mise à jour")
    plt.ylabel("Entropie")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
