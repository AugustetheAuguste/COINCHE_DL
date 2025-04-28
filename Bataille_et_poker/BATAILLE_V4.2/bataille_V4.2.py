import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import datetime

# -----------------------------------------------------------
# Paramètres globaux du jeu et PPO
# -----------------------------------------------------------
NB_CARTES = 52           # Deck de 52 cartes uniques
NB_CARTES_AGENT = 26     # Chaque joueur reçoit 26 cartes
NB_TOURS = 26            # Nombre de tours total dans une partie (pour inclure la mémoire complète)
SUITS = ["coeur", "trèfle", "carreau", "pique"]

# Hyperparamètres PPO
GAMMA = 0.99
CLIP_EPSILON = 0.2
BATCH_EPISODES = 8       # Nombre d'épisodes collectés avant mise à jour
NUM_EPOCHS = 10          # Nombre d'epochs d'optimisation sur le batch
MINIBATCH_SIZE = 8       # Taille du mini-batch lors de l'optimisation
LR = 3e-4                # Taux d'apprentissage
VF_COEF = 0.5            # Coefficient de la loss de valeur
ENTROPY_COEF = 0.01      # Coefficient bonus d'entropie

# Dossier de sauvegarde du meilleur modèle
SAVE_PATH = "./best_model.pt"

# -----------------------------------------------------------
# Fonction d'encodage d'une carte
# -----------------------------------------------------------
def encode_card(card):
    """
    Encode une carte sous la forme d'un vecteur de dimension 5 :
      - La première valeur représente le rang normalisé (de 0 à 1), où 2 → 0.0 et As (14) → 1.0.
      - Les 4 valeurs suivantes correspondent à un encodage one-hot de la couleur,
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
        # Création d'un deck de 52 cartes uniques (rangs 2 à 14, 14 = As)
        self.deck = [(rank, suit) for suit in SUITS for rank in range(2, 15)]
        random.shuffle(self.deck)
        # Distribution : première moitié pour l'adversaire, deuxième moitié pour l'agent
        self.adversaire_main = self.deck[:NB_CARTES_AGENT]
        self.agent_main = self.deck[NB_CARTES_AGENT:]
        self.tour = 0
        self.nouvelle_manche()
        return self.get_observation()
    
    def nouvelle_manche(self):
        """
        L'adversaire joue une carte aléatoirement depuis sa main.
        """
        if len(self.adversaire_main) > 0:
            idx = random.randrange(len(self.adversaire_main))
            self.current_adversaire_carte = self.adversaire_main.pop(idx)
        else:
            self.current_adversaire_carte = None
    
    def get_observation(self):
        """
        Renvoie un dictionnaire contenant :
          - 'hand' : un tableau de forme (26, 5) pour la main de l'agent,
          - 'adv'  : un tableau de forme (5,) pour la carte de l'adversaire.
        """
        hand_obs = []
        for card in self.agent_main:
            hand_obs.append(encode_card(card))
        # Compléter avec des zéros si l'agent a moins de 26 cartes
        while len(hand_obs) < NB_CARTES_AGENT:
            hand_obs.append(np.zeros(5, dtype=np.float32))
        hand_obs = np.stack(hand_obs)  # forme (26, 5)
        adv_obs = encode_card(self.current_adversaire_carte)  # forme (5,)
        return {'hand': hand_obs.astype(np.float32), 'adv': adv_obs.astype(np.float32)}
    
    def get_valid_action_mask(self):
        """
        Retourne un masque (taille 26) indiquant les positions jouables dans la main de l'agent.
        - Seules les positions non vides sont initialement marquées comme valides.
        - Si l'agent possède au moins une carte de la même couleur que celle de l'adversaire,
          seules ces positions seront validées.
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
        return mask
    
    def step(self, action):
        """
        L'agent joue la carte à l'index donné dans sa main.
        Règles :
          - Si la carte jouée est de la même couleur que celle de l'adversaire :
              * Si son rang est supérieur → reward = +1
              * Sinon → reward = -1
          - Sinon (cas où l'agent doit se défausser) → reward = -1.
        La carte jouée est retirée de la main.
        """
        done = False
        if action < 0 or action >= NB_CARTES_AGENT or self.agent_main[action] is None:
            reward = -1.0
        else:
            carte_agent = self.agent_main[action]
            adv_carte = self.current_adversaire_carte
            if carte_agent[1] == adv_carte[1]:
                if carte_agent[0] > adv_carte[0]:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                reward = -1.0
            # Retirer la carte jouée
            self.agent_main[action] = None
        
        self.tour += 1
        if self.tour >= self.nb_tours:
            done = True
        else:
            self.nouvelle_manche()
        obs = self.get_observation()
        return obs, reward, done, {}
    
    def render(self):
        pass  # On ne fait plus de logs ni de print ici.

# -----------------------------------------------------------
# Réseau Actor-Critic récurrent avec GRU
# -----------------------------------------------------------
class RecurrentActorCriticNetwork(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        """
        Chaque observation est constituée de :
         - hand : (26,5) → aplati en 130 dimensions
         - adv  : (5,)   → concaténé pour obtenir 135 dimensions
        """
        super(RecurrentActorCriticNetwork, self).__init__()
        self.input_dim = NB_CARTES_AGENT * 5 + 5  # 26*5 + 5 = 135
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, hand_seq, adv_seq, h0=None):
        """
        hand_seq: (batch, seq_len, 26, 5)
        adv_seq:  (batch, seq_len, 5)
        h0: (1, batch, hidden_dim)
        """
        batch, seq_len, _, _ = hand_seq.shape
        # Aplatir la main pour chaque pas de temps : (batch, seq_len, 130)
        hand_flat = hand_seq.view(batch, seq_len, -1)
        x = torch.cat([hand_flat, adv_seq], dim=2)  # (batch, seq_len, 135)
        x = F.relu(self.fc1(x))  # (batch, seq_len, hidden_dim)
        if h0 is None:
            h0 = torch.zeros(1, batch, x.size(-1)).to(x.device)
        gru_out, h_n = self.gru(x, h0)  # gru_out: (batch, seq_len, hidden_dim)
        logits = self.policy_head(gru_out)  # (batch, seq_len, action_dim)
        values = self.value_head(gru_out).squeeze(-1)  # (batch, seq_len)
        return logits, values, h_n

# -----------------------------------------------------------
# Agent PPO récurrent avec GRU
# -----------------------------------------------------------
class PPOAgent:
    def __init__(self, hidden_dim=128, lr=LR):
        self.model = RecurrentActorCriticNetwork(hidden_dim, NB_CARTES_AGENT)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_epsilon = CLIP_EPSILON
        self.gamma = GAMMA
        self.vf_coef = VF_COEF
        self.entropy_coef = ENTROPY_COEF
        self.hidden_dim = hidden_dim

    def select_action(self, obs, valid_mask, h):
        """
        Sélectionne une action pour un pas de temps donné en mode récurrent.
        obs: dictionnaire {'hand': (26,5), 'adv': (5,)}
        h: état caché (1,1,hidden_dim)
        Retourne l'action, log_prob, la valeur et le nouvel état caché.
        """
        # Ajouter des dimensions batch et seq_len = 1
        hand = torch.from_numpy(obs['hand']).float().unsqueeze(0).unsqueeze(0)  # (1,1,26,5)
        adv = torch.from_numpy(obs['adv']).float().unsqueeze(0).unsqueeze(0)    # (1,1,5)
        logits, value, h_new = self.model(hand, adv, h)
        logits = logits.squeeze(0).squeeze(0)  # (NB_CARTES_AGENT,)
        value = value.squeeze(0).squeeze(0).item()
        mask_tensor = torch.from_numpy(valid_mask).float()  # (NB_CARTES_AGENT,)
        masked_logits = logits + (1 - mask_tensor) * (-1e8)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob, value, h_new

    def compute_returns_and_advantages(self, rewards, values, dones):
        """
        Calcule les retours G_t et les avantages A_t = G_t - V(s_t) pour une trajectoire.
        """
        returns = []
        G = 0
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
        Mise à jour PPO sur un batch d'épisodes collectés.
        Ici, chaque épisode est une séquence de NB_TOURS pas de temps.
        Le batch est un dictionnaire contenant 'hand', 'adv', 'actions', 'log_probs', 'returns', 'advantages'
        Les tenseurs ont pour forme :
          - hand: (batch, seq_len, 26, 5)
          - adv:  (batch, seq_len, 5)
          - actions, log_probs, returns, advantages: (batch, seq_len)
        """
        hand_seq = torch.tensor(np.array(batch['hand']), dtype=torch.float32)  # (N, seq_len, 26, 5)
        adv_seq = torch.tensor(np.array(batch['adv']), dtype=torch.float32)    # (N, seq_len, 5)
        actions = torch.tensor(batch['actions'], dtype=torch.long)             # (N, seq_len)
        old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32)  # (N, seq_len)
        returns = torch.tensor(batch['returns'], dtype=torch.float32)          # (N, seq_len)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32)    # (N, seq_len)

        total_samples = hand_seq.shape[0]  # nombre d'épisodes
        policy_losses = []
        value_losses = []
        entropies = []

        indices = np.arange(total_samples)
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, total_samples, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_idx = indices[start:end]
                mb_hand = hand_seq[mb_idx]     
                mb_adv = adv_seq[mb_idx]       
                mb_actions = actions[mb_idx]   
                mb_old_log_probs = old_log_probs[mb_idx] 
                mb_returns = returns[mb_idx]   
                mb_advantages = advantages[mb_idx]  

                logits, values, _ = self.model(mb_hand, mb_adv)
                mb, seq_len, _ = logits.shape
                logits = logits.view(mb * seq_len, -1)
                dist = torch.distributions.Categorical(logits=logits)
                mb_actions_flat = mb_actions.view(-1)
                new_log_probs = dist.log_prob(mb_actions_flat)
                entropy = dist.entropy().mean()

                mb_old_log_probs = mb_old_log_probs.view(-1)
                mb_advantages = mb_advantages.view(-1)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = values.view(-1)
                mb_returns = mb_returns.view(-1)
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
        Collecte des épisodes complets (séquences de NB_TOURS pas de temps).
        Retourne un batch et le reward moyen sur ces épisodes.
        Pour chaque épisode, on stocke les observations sous forme séquentielle.
        """
        batch = {
            'hand': [],
            'adv': [],
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
            h = torch.zeros(1, 1, self.hidden_dim)
            ep_hand = []
            ep_adv = []
            ep_actions = []
            ep_log_probs = []
            ep_rewards_list = []
            ep_dones = []
            ep_values = []
            while not done:
                valid_mask = env.get_valid_action_mask()
                ep_hand.append(obs['hand'])
                ep_adv.append(obs['adv'])
                action, log_prob, value, h = self.select_action(obs, valid_mask, h)
                ep_actions.append(action)
                ep_log_probs.append(log_prob.item())
                ep_values.append(value)
                obs, reward, done, _ = env.step(action)
                ep_rewards_list.append(reward)
                ep_dones.append(done)
            episode_rewards.append(sum(ep_rewards_list))
            returns, advantages = self.compute_returns_and_advantages(ep_rewards_list, ep_values, ep_dones)
            batch['hand'].append(np.array(ep_hand))
            batch['adv'].append(np.array(ep_adv))
            batch['actions'].append(np.array(ep_actions))
            batch['log_probs'].append(np.array(ep_log_probs))
            batch['rewards'].append(np.array(ep_rewards_list))
            batch['dones'].append(np.array(ep_dones))
            batch['values'].append(np.array(ep_values))
            batch['returns'].append(returns.numpy())
            batch['advantages'].append(advantages.numpy())
        avg_reward = np.mean(episode_rewards)
        return batch, avg_reward

# -----------------------------------------------------------
# Boucle d'entraînement principale
# -----------------------------------------------------------
def main():
    # Création du dossier "plots" si inexistant
    os.makedirs("plots", exist_ok=True)

    env = BatailleEnv()
    agent = PPOAgent(hidden_dim=128, lr=LR)
    
    num_updates = 500  # Nombre d'itérations de mise à jour
    rewards_history = []
    policy_losses_history = []
    value_losses_history = []
    entropy_history = []
    
    best_reward = -float('inf')
    
    for update in range(num_updates):
        batch, avg_reward = agent.collect_trajectories(env, batch_episodes=BATCH_EPISODES)
        rewards_history.append(avg_reward)
        avg_policy_loss, avg_value_loss, avg_entropy = agent.update(batch)
        policy_losses_history.append(avg_policy_loss)
        value_losses_history.append(avg_value_loss)
        entropy_history.append(avg_entropy)
        
        # Sauvegarde du meilleur modèle
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(agent.model.state_dict(), SAVE_PATH)
        
        # Tous les 100 updates, on génère et on enregistre les graphes
        if (update + 1) % 100 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plots/plot_update_{update+1}_{timestamp}.png"
            plt.figure(figsize=(14,6))

            # Reward moyen
            plt.subplot(1,3,1)
            plt.plot(rewards_history)
            plt.title("Reward moyen par mise à jour")
            plt.xlabel("Mise à jour")
            plt.ylabel("Reward moyen")

            # Policy Loss et Value Loss
            plt.subplot(1,3,2)
            plt.plot(policy_losses_history, label="Policy Loss")
            plt.plot(value_losses_history, label="Value Loss")
            plt.title("Loss par mise à jour")
            plt.xlabel("Mise à jour")
            plt.ylabel("Loss")
            plt.legend()

            # Entropie
            plt.subplot(1,3,3)
            plt.plot(entropy_history)
            plt.title("Entropie moyenne par mise à jour")
            plt.xlabel("Mise à jour")
            plt.ylabel("Entropie")

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

if __name__ == "__main__":
    main()
