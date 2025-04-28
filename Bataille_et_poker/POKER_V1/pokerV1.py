#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V2 - Implémentation de Leduc Hold’em en self‑play avec PPO + LSTM et monitoring détaillé

Améliorations par rapport à la version V1 :
- Conversion optimisée des listes en tenseurs (utilisation de np.array() pour éviter les warnings PyTorch).
- Logging détaillé : enregistrement de la récompense, de la loss et du taux de victoire dans un fichier JSONL.
- Fonction d'évaluation périodique pour mesurer le win rate sur un ensemble de parties.
- Plus de commentaires pour clarifier le fonctionnement et aider à comprendre l’ensemble du pipeline.

Choix techniques :
- Environnement personnalisé (héritage de gym.Env) afin de maîtriser complètement les règles de Leduc Hold’em.
- Encodage one‑hot pour les cartes privées et publiques (jeu à 6 cartes).
- Traitement complet de l’historique par le LSTM (pour conserver l’ordre temporel).
- Masquage d’actions appliqué directement aux logits pour garantir que seules les actions légales soient prises.
- Algorithme PPO avec calcul du GAE (Generalized Advantage Estimation) pour une mise à jour stable.
- Monitoring via fichiers JSONL et plots en local avec Matplotlib.

Sources et inspirations :
  - PPO d’OpenAI, CleanRL pour la structure PPO (cite_openai_ppo, cite_cleanrl).
  - Remarques sur la conversion de liste de numpy.ndarray en tenseurs issues des warnings PyTorch cite_pytorch_conversion.
  
Auteur : [Votre Nom]
Date : [Date actuelle]
"""

import gym
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

##############################
# Environnement Leduc Hold’em
##############################

class LeducHoldemEnv(gym.Env):
    """
    Environnement simplifié pour Leduc Hold’em.
    
    Règles simplifiées :
      - Deck de 6 cartes (2 couleurs x 3 rangs).
      - Chaque joueur reçoit une carte privée ; une carte commune est révélée après la phase pre-flop.
      - Deux phases de pari : phase 0 (pre-flop) et phase 1 (post-flop).
      - Actions simplifiées : 0 = Check/Call, 1 = Bet/Fold.
      - Un masquage d’actions est fourni via get_valid_actions().
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LeducHoldemEnv, self).__init__()
        self.deck = list(range(6))
        self.private_cards = [None, None]
        self.public_card = None
        
        self.history = []          # Historique des actions (liste d'entiers)
        self.max_history = 4       # Nombre max d'actions à garder dans l’historique
        
        self.action_space = gym.spaces.Discrete(2)  # 2 actions possibles
        obs_dim = 6 + 6 + (self.max_history * 2)      # private one-hot, public one-hot, historique encodé one-hot
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        self.current_player = 0   # Indique quel joueur doit agir
        self.phase = 0            # 0: pre-flop, 1: post-flop
        self.done = False
        self.winner = None

    def reset(self):
        # Réinitialisation des variables de jeu
        self.deck = list(range(6))
        random.shuffle(self.deck)
        self.private_cards[0] = self.deck.pop()
        self.private_cards[1] = self.deck.pop()
        self.public_card = None
        self.history = []
        self.current_player = 0
        self.phase = 0
        self.done = False
        self.winner = None
        return self._get_obs(), self.get_valid_actions()
    
    def _one_hot_card(self, card):
        vec = np.zeros(6, dtype=np.float32)
        if card is not None:
            vec[card] = 1.0
        return vec

    def _one_hot_history(self):
        hist = np.zeros(self.max_history * 2, dtype=np.float32)
        for i, a in enumerate(self.history[-self.max_history:]):
            one_hot = np.zeros(2, dtype=np.float32)
            one_hot[a] = 1.0
            hist[i*2:(i+1)*2] = one_hot
        return hist

    def _get_obs(self):
        private = self._one_hot_card(self.private_cards[self.current_player])
        public = self._one_hot_card(self.public_card) if (self.phase == 1 and self.public_card is not None) else np.zeros(6, dtype=np.float32)
        hist = self._one_hot_history()
        return np.concatenate([private, public, hist]).astype(np.float32)

    def get_valid_actions(self):
        valid = np.array([1, 1], dtype=np.float32)
        if self.phase == 0:
            if 1 in self.history:
                valid[1] = 0
        else:
            if 1 in self.history:
                valid[1] = 0  # Simplification : si un bet a déjà été fait, seule l’action call est autorisée
        return valid

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode terminé. Veuillez appeler reset()")
        
        valid = self.get_valid_actions()
        if valid[action] == 0:
            # Action illégale : pénalité et fin d’épisode
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done, {"illegal": True}, valid

        self.history.append(action)

        # Passage à la phase suivante ou fin d'épisode
        if len(self.history) >= 2:
            if self.phase == 0:
                if self.public_card is None and len(self.deck) > 0:
                    self.public_card = self.deck.pop()
                self.phase = 1
                self.history = []  # réinitialiser l’historique pour le second round
            else:
                self.done = True
                # Règle de victoire simplifiée
                if 1 in self.history:
                    self.winner = self.current_player
                else:
                    curr_private = self.private_cards[self.current_player]
                    opp_private = self.private_cards[1 - self.current_player]
                    if curr_private == self.public_card and opp_private != self.public_card:
                        self.winner = self.current_player
                    elif opp_private == self.public_card and curr_private != self.public_card:
                        self.winner = 1 - self.current_player
                    else:
                        self.winner = random.choice([0, 1])
        # Alterner les rôles
        self.current_player = 1 - self.current_player
        
        if self.done:
            # La récompense est donnée par rapport au joueur qui vient d'agir (après le changement)
            reward = 1 if self.winner == self.current_player else -1
        else:
            reward = 0
        
        return self._get_obs(), reward, self.done, {}, self.get_valid_actions()

    def render(self, mode="human"):
        print(f"Joueur courant: {self.current_player}")
        print(f"Cartes privées: {self.private_cards}")
        print(f"Carte publique: {self.public_card}")
        print(f"Historique: {self.history}")
        print(f"Phase: {self.phase}")
        print(f"Actions valides: {self.get_valid_actions()}")

##########################################
# Modèle Actor-Critic avec LSTM (PyTorch)
##########################################

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, hidden_state=None, action_mask=None):
        # Passage par la couche FC
        x = self.relu(self.fc1(x))
        # Ajouter la dimension temporelle si nécessaire
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor(lstm_out)
        # Application du masquage sur les logits pour ignorer les actions illégales
        if action_mask is not None:
            neg_inf = -1e9 * (1 - action_mask)
            logits = logits + neg_inf
        value = self.critic(lstm_out)
        return logits, value, new_hidden

###########################################
# Fonctions utilitaires : GAE et PPO Update
###########################################

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]  # Valeur terminale
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns

# Hyperparamètres PPO
LR = 3e-4
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
PPO_EPOCHS = 4
BATCH_SIZE = 32

def ppo_update(model, optimizer, observations, actions, log_probs_old, returns, advantages):
    # Conversion optimisée : conversion de liste en np.array pour éviter les warnings PyTorch (cite_pytorch_conversion)
    observations = torch.tensor(np.array(observations), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.long)
    log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32)
    returns = torch.tensor(np.array(returns), dtype=torch.float32)
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
    
    dataset_size = observations.shape[0]
    for _ in range(PPO_EPOCHS):
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_obs = observations[i:i+BATCH_SIZE]
            batch_actions = actions[i:i+BATCH_SIZE]
            batch_log_probs_old = log_probs_old[i:i+BATCH_SIZE]
            batch_returns = returns[i:i+BATCH_SIZE]
            batch_advantages = advantages[i:i+BATCH_SIZE]
            
            logits, values, _ = model(batch_obs, None, None)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(batch_actions)
            
            ratio = torch.exp(log_probs - batch_log_probs_old)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = ((batch_returns - values.squeeze(-1)) ** 2).mean()
            entropy_loss = dist.entropy().mean()
            
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
    return loss.item()

############################################
# Fonctions de monitoring et d'évaluation
############################################

def plot_training(rewards, losses, win_rates):
    if not os.path.exists("plot"):
        os.makedirs("plot")
    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Récompense par épisode")
    plt.savefig("plot/rewards.png")
    plt.close()
    
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss par épisode")
    plt.savefig("plot/loss.png")
    plt.close()
    
    plt.figure()
    plt.plot(win_rates)
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Taux de victoire (win rate)")
    plt.savefig("plot/win_rate.png")
    plt.close()

def evaluate_agent(model, env, num_episodes=100):
    wins = 0
    total_reward = 0
    for _ in range(num_episodes):
        obs, valid_mask = env.reset()
        done = False
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0)
            logits, _, _ = model(obs_tensor, None, valid_mask_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            obs, reward, done, _, valid_mask = env.step(action)
        # On considère la récompense finale pour déterminer la victoire (ici +1 ou -1)
        total_reward += reward
        if reward > 0:
            wins += 1
    win_rate = wins / num_episodes
    avg_reward = total_reward / num_episodes
    return win_rate, avg_reward

###############################
# Boucle d'entraînement principale
###############################

def train():
    env = LeducHoldemEnv()
    obs_example, _ = env.reset()
    input_dim = obs_example.shape[0]
    num_actions = env.action_space.n
    
    model = ActorCritic(input_dim, hidden_dim=128, num_actions=num_actions)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    num_episodes = 100_000  # Augmenter le nombre d'épisodes pour voir de vraies améliorations
    log_file = "training_log.jsonl"
    if os.path.exists(log_file):
        os.remove(log_file)
    
    all_rewards = []
    all_losses = []
    all_win_rates = []
    
    # Pour évaluer la performance sur une fenêtre glissante
    eval_interval = 50
    
    for episode in range(num_episodes):
        obs, valid_mask = env.reset()
        done = False
        episode_rewards = 0

        observations = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        masks = []
        
        # Self-play : on laisse l'agent jouer contre son adversaire
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            valid_mask_tensor = torch.tensor(valid_mask, dtype=torch.float32).unsqueeze(0)
            logits, value, _ = model(obs_tensor, None, valid_mask_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
            
            observations.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.item())
            
            obs, reward, done, info, valid_mask = env.step(action)
            rewards.append(reward)
            masks.append(0 if done else 1)
            episode_rewards += reward
        
        advantages, returns = compute_gae(rewards, values, masks)
        loss_value = ppo_update(model, optimizer, observations, actions, log_probs, returns, advantages)
        
        all_rewards.append(episode_rewards)
        all_losses.append(loss_value)
        
        # Évaluer périodiquement le modèle et calculer le win rate
        if episode % eval_interval == 0 and episode > 0:
            win_rate, avg_reward = evaluate_agent(model, env, num_episodes=100)
            all_win_rates.append(win_rate)
            print(f"Épisode {episode}, Reward: {episode_rewards:.2f}, Loss: {loss_value:.4f}, WinRate: {win_rate:.2f}, AvgRewardEval: {avg_reward:.2f}")
        else:
            # Si aucune évaluation, on réutilise la dernière valeur connue
            if all_win_rates:
                current_win_rate = all_win_rates[-1]
            else:
                current_win_rate = 0.0

        # Sauvegarder les logs pour chaque épisode
        log_data = {
            "episode": episode,
            "reward": episode_rewards,
            "loss": loss_value,
            "win_rate": all_win_rates[-1] if all_win_rates else None
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")
    
    # Sauvegarde du modèle final
    torch.save(model.state_dict(), "actor_critic_model_v2.pt")
    
    # Génération des plots
    plot_training(all_rewards, all_losses, all_win_rates)
    
    # Évaluation finale et impression des résultats
    final_win_rate, final_avg_reward = evaluate_agent(model, env, num_episodes=200)
    print(f"Évaluation finale sur 200 épisodes => Win Rate: {final_win_rate:.2f}, Avg Reward: {final_avg_reward:.2f}")

if __name__ == "__main__":
    train()
