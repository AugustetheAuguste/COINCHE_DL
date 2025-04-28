#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ce script permet de jouer contre l’agent entraîné (actor_critic_model_v2.pt) dans un
environnement simplifié de Leduc Hold’em. Une interface graphique simple est implémentée
avec Tkinter pour afficher l’état du jeu et permettre au joueur humain de choisir des actions.

Les choix importants :
  - L’environnement est défini de manière personnalisée pour maîtriser l’encodage des cartes,
    l’historique, le masquage des actions et la dynamique de passage de tour.
  - Le modèle Actor‑Critic avec LSTM est exactement celui utilisé pour l’entraînement.
  - Pour le masquage, on applique directement sur les logits afin de garantir que seules
    les actions légales soient jouées.
  - L’interface utilise des boutons pour permettre au joueur de choisir entre les actions 0 et 1.
  
Auteur : [Votre Nom]
Date : [Date actuelle]
"""

import tkinter as tk
from tkinter import messagebox
import torch
import numpy as np
import random
import os
import gym
from torch.distributions import Categorical
import torch.nn as nn

##############################
# Environnement Leduc Hold’em
##############################

class LeducHoldemEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(LeducHoldemEnv, self).__init__()
        # Définir le deck avec 6 cartes (par exemple, indexées de 0 à 5)
        self.deck = list(range(6))
        self.private_cards = [None, None]  # Une carte pour chaque joueur
        self.public_card = None            # Carte commune (non révélée au début)
        
        self.history = []         # Historique des actions (liste d'entiers)
        self.max_history = 4      # Nombre maximum d'actions à conserver pour l’historique
        
        # Deux actions possibles : 0 = Check/Call, 1 = Bet/Fold
        self.action_space = gym.spaces.Discrete(2)
        # Observation composée de :
        #   - Carte privée (6 dimensions one-hot)
        #   - Carte publique (6 dimensions one-hot)
        #   - Historique (max_history*2 dimensions, one-hot pour chaque action)
        obs_dim = 6 + 6 + (self.max_history * 2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Indique le joueur courant : 0 ou 1 (on suppose que le joueur humain sera 0)
        self.current_player = 0
        self.phase = 0  # 0: pre-flop, 1: post-flop
        self.done = False
        self.winner = None

    def reset(self):
        self.deck = list(range(6))
        random.shuffle(self.deck)
        # Distribution des cartes privées
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
            # En pre-flop, si une action 1 a déjà été jouée, interdiction de rejouer une mise
            if 1 in self.history:
                valid[1] = 0
        else:
            # En post-flop, on interdit également l'action "Bet" si déjà jouée (simplification)
            if 1 in self.history:
                valid[1] = 0
        return valid

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode terminé. Appeler reset()")
        
        valid = self.get_valid_actions()
        if valid[action] == 0:
            # Action illégale : pénalité et terminaison de la partie
            reward = -10
            self.done = True
            return self._get_obs(), reward, self.done, {"illegal": True}, valid

        # Enregistrer l'action dans l'historique
        self.history.append(action)

        # Gestion de la transition entre les phases de jeu
        if len(self.history) >= 2:
            if self.phase == 0:
                # Révéler la carte commune et passer en phase post-flop
                if self.public_card is None and len(self.deck) > 0:
                    self.public_card = self.deck.pop()
                self.phase = 1
                self.history = []  # Réinitialiser l’historique pour le second round
            else:
                self.done = True
                # Règle de victoire simplifiée :
                # Si une action "Bet" apparaît, le joueur courant l’emporte.
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
        # Changement de tour
        self.current_player = 1 - self.current_player
        
        if self.done:
            reward = 1 if self.winner == self.current_player else -1
        else:
            reward = 0
        
        return self._get_obs(), reward, self.done, {}, self.get_valid_actions()
    
    def render(self, mode="human"):
        # Méthode de rendu pour le debug, ici nous utilisons un affichage textuel
        print(f"Joueur courant: {self.current_player}")
        print(f"Cartes privées: {self.private_cards}")
        print(f"Carte publique: {self.public_card}")
        print(f"Historique: {self.history}")
        print(f"Phase: {self.phase}")
        print(f"Actions valides: {self.get_valid_actions()}")

###########################################
# Modèle Actor‑Critic avec LSTM (PyTorch)
###########################################

class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # LSTM d'une seule couche pour capturer l’historique temporel
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Branche de l’Actor : produit les logits pour chaque action
        self.actor = nn.Linear(hidden_dim, num_actions)
        # Branche du Critic : estime la valeur d’état
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, hidden_state=None, action_mask=None):
        x = self.relu(self.fc1(x))
        # Ajouter une dimension temporelle s’il n’en existe pas déjà
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(x, hidden_state)
        lstm_out = lstm_out.squeeze(1)
        logits = self.actor(lstm_out)
        # Appliquer le masquage : attribuer -1e9 aux actions illégales
        if action_mask is not None:
            neg_inf = -1e9 * (1 - action_mask)
            logits = logits + neg_inf
        value = self.critic(lstm_out)
        return logits, value, new_hidden

###########################################
# Fonction utilitaire pour afficher l'état
###########################################

def state_to_text(env, human_player=0):
    """
    Génère un texte résumant l'état du jeu de manière simple pour l'interface.
    On affiche la carte privée du joueur humain, la carte publique, l'historique et le tour de jeu.
    """
    text = ""
    # Afficher la carte privée du joueur humain
    if human_player == 0:
        text += f"Votre carte privée: {env.private_cards[0]}\n"
    else:
        text += f"Votre carte privée: {env.private_cards[1]}\n"
    text += f"Carte publique: {env.public_card if env.public_card is not None else 'Non révélée'}\n"
    text += f"Historique: {env.history}\n"
    text += f"Phase: {'Pre-flop' if env.phase==0 else 'Post-flop'}\n"
    if env.done:
        if env.winner is None:
            text += "Match nul ou non déterminé.\n"
        else:
            if env.winner == human_player:
                text += "Vous avez gagné !\n"
            else:
                text += "Vous avez perdu.\n"
    else:
        if env.current_player == human_player:
            text += "C'est votre tour.\n"
        else:
            text += "Tour de l'adversaire...\n"
    return text

###########################################
# Interface Graphique avec Tkinter
###########################################

class GameGUI:
    def __init__(self, root, env, model, human_player=0):
        self.root = root
        self.env = env
        self.model = model
        self.human_player = human_player
        
        # Réinitialiser l'environnement au début du jeu
        self.obs, self.valid_mask = self.env.reset()
        
        # Cadre principal
        self.frame = tk.Frame(root)
        self.frame.pack(padx=10, pady=10)
        
        # Zone d'affichage de l'état
        self.state_label = tk.Label(self.frame, text=state_to_text(self.env, human_player),
                                     font=("Helvetica", 12), justify="left")
        self.state_label.pack()
        
        # Cadre contenant les boutons d'action
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(pady=10)
        
        self.btn_action0 = tk.Button(self.button_frame, text="Action 0 (Check/Call)",
                                       command=lambda: self.human_move(0))
        self.btn_action0.grid(row=0, column=0, padx=5)
        
        self.btn_action1 = tk.Button(self.button_frame, text="Action 1 (Bet/Fold)",
                                       command=lambda: self.human_move(1))
        self.btn_action1.grid(row=0, column=1, padx=5)
        
        # Bouton de redémarrage (désactivé jusqu'à la fin d'une partie)
        self.btn_restart = tk.Button(self.frame, text="Nouvelle Partie", command=self.restart_game)
        self.btn_restart.pack(pady=5)
        self.btn_restart.config(state=tk.DISABLED)
        
        self.update_ui()
        # Si c'est au tour de l'adversaire dès le départ, lancer l'action de l'agent après un court délai
        self.root.after(500, self.check_turn)
    
    def update_ui(self):
        self.state_label.config(text=state_to_text(self.env, self.human_player))
    
    def human_move(self, action):
        # Ne rien faire si ce n'est pas le tour du joueur humain ou si l'épisode est terminé
        if self.env.current_player != self.human_player or self.env.done:
            return
        # Vérifier la validité de l'action
        if self.valid_mask[action] == 0:
            messagebox.showinfo("Action Invalide", "Cette action n'est pas valide !")
            return
        
        # Appliquer l'action du joueur humain
        self.obs, reward, done, info, self.valid_mask = self.env.step(action)
        self.update_ui()
        if done:
            self.end_game()
        else:
            # Lancer le tour de l'agent après un court délai
            self.root.after(500, self.check_turn)
    
    def check_turn(self):
        # Si l'épisode est terminé, afficher le résultat
        if self.env.done:
            self.end_game()
            return
        # Si ce n'est pas le tour du joueur humain, l'agent doit jouer
        if self.env.current_player != self.human_player:
            self.agent_move()
    
    def agent_move(self):
        # L'agent choisit son action en utilisant le modèle chargé
        obs_tensor = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        valid_mask_tensor = torch.tensor(self.valid_mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = self.model(obs_tensor, None, valid_mask_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample().item()
        self.obs, reward, done, info, self.valid_mask = self.env.step(action)
        self.update_ui()
        if done:
            self.end_game()
    
    def end_game(self):
        self.update_ui()
        self.btn_restart.config(state=tk.NORMAL)
        messagebox.showinfo("Fin de Partie", "La partie est terminée !")
    
    def restart_game(self):
        self.obs, self.valid_mask = self.env.reset()
        self.update_ui()
        self.btn_restart.config(state=tk.DISABLED)
        self.root.after(500, self.check_turn)

###########################################
# Programme Principal
###########################################

def main():
    # Créer l'environnement et charger le modèle entraîné
    env = LeducHoldemEnv()
    obs, _ = env.reset()
    input_dim = obs.shape[0]
    num_actions = env.action_space.n
    
    model = ActorCritic(input_dim, hidden_dim=128, num_actions=num_actions)
    model_path = "actor_critic_model_v2.pt"
    if os.path.exists(model_path):
        # Chargement sur le CPU (adaptable selon vos ressources)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        print("Modèle introuvable : vérifiez que 'actor_critic_model_v2.pt' existe.")
        return
    
    root = tk.Tk()
    root.title("Jouer contre l'Agent - Leduc Hold'em")
    game = GameGUI(root, env, model, human_player=0)
    root.mainloop()

if __name__ == "__main__":
    main()
