from random import randint
import gym
import numpy as np
from gym import spaces
from game import CoinceGame
from utils import Card, Rank, Suit

class CoincheEnv(gym.Env):
    def __init__(self):
        super(CoincheEnv, self).__init__()
        self.game = CoinceGame()

        self.suit = ["SPADES", "HEARTS", "DIAMONDS", "CLUBS", "No_ASSET","Full_ASSET"]
        self.rank = ["7", "8", "9", "10", "J", "Q", "K", "A"]
        self.current_player = 0
        self.history = []
        self.flag = False
        self.points_team = [0,0]

        # Définition des espaces d'observation et d'action
        self.observation_space = spaces.Box(low=0, high=1, shape=(48,), dtype=np.float32)  # Représentation des cartes en main
        self.action_space = spaces.Discrete(8)  # Nombre total d'actions possibles (annonces, jeux de cartes, etc.)

        self.reset()

    def reset(self):
        """Réinitialise le jeu et retourne l'état initial."""
        self.game = CoinceGame()
        self.game.shuffle_deck()
        self.game.deal_card(self.game.current_player, [3, 2, 3])
        self.set_bid()
        self.current_player = self.game.current_player
        self.game.table.set_current_player(self.current_player)
        self.state = self.get_observation()
        return self.state

    def set_bid(self):
        player = randint(0, 3)
        couleur = randint(0, 5)
        self.game.table.announce((80,self.suit[couleur]), player)

    def get_observation(self):
        """Retourne une observation complète en encodant mieux la main du joueur."""
        
        observation = np.zeros(48)  # 8 (main) + 1 (atout) + 1 (contrat) + 1 (coinche) + 1 (tour) + 4 (cartes posées) + 28 (plis précédents)

        current_player = self.current_player

        # 1️⃣ Encodage de la main (8 entrées, index des cartes ou -1 si absent)
        cards_in_hand = self.game.players[current_player].get_card()
        for i in range(8):
            if i < len(cards_in_hand):
                observation[i] = self.card_to_index(cards_in_hand[i]) / 52.0  # Normalisé
            else:
                observation[i] = -1  # Placeholder pour cartes manquantes

        # 2️⃣ Atout (1 entrée, normalisé)
        observation[8] = self.suit.index(self.game.table.get_current_asset())

        # 3️⃣ Contrat (1 entrée, normalisé)
        observation[9] = (self.game.table.get_current_bid().get_points() - 80) 

        # 4️⃣ Coinche (1 entrée)
        observation[10] = self.game.table.get_current_bid().get_coef()

        # 5️⃣ Tour actuel (1 entrée, normalisé)
        observation[11] = (8 - len(self.game.players[0].get_card())) / 8.0  

        # 6️⃣ Cartes posées (4 entrées, index normalisés ou -1 si vide)
        for i, card in enumerate(self.game.table.get_card()):
            observation[12 + i] = self.card_to_index(card) / 52.0 
        for i in range(len(self.game.table.get_card()), 4):
            observation[12 + i] = -1  # Cartes non jouées

        # 7️⃣ Plis précédents (32 entrées max, 7 derniers plis)
        pli_offset = 16
        last_plis = self.game.table.get_all_cards()  # On prend les 7 derniers plis max
        for i, card in enumerate(last_plis):
            index = pli_offset + i
            observation[index] = self.card_to_index(card) / 52.0  
        for i in range(len(last_plis), 32):  # Remplir les espaces vides
            observation[pli_offset + i] = -1

        return observation



    def card_to_index(self, card):
        """Convertit une carte en un index unique dans l'observation."""
        suits = {"HEARTS": 0, "DIAMONDS": 1, "CLUBS": 2, "SPADES": 3}
        ranks = {"7": 0, "8": 1, "9": 2, "10": 3, "J": 4, "Q": 5, "K": 6, "A": 7}
        return suits[card.suit.name] * 8 + ranks[card.rank.value]

    def step(self, action):

        """Exécute une action et met à jour l'état du jeu."""
        if self.game.game_finish() and self.flag:
            reward = self.points_team[self.current_player%2]
            self.current_player = (self.current_player + 1) % 4
            if self.game.table.current_player == self.current_player:
                self.flag = False
                obs = self.history[-1][0]
                self.history.clear() 
                return obs, reward, True, {}
            return self.history[-1][0], reward, False, {}
        else:
        
            legal_cards = self.game.get_legal_cards(self.game.players[self.current_player])
            
            # Vérifier que l'action correspond bien à une carte légale
            if action >= len(self.game.players[self.current_player].get_card()):
                reward = -10  # Pénalité pour action hors limite
                return self.get_observation(), reward, False, {}
            
            chosen_card = self.game.players[self.current_player].get_card()[action]

            if chosen_card not in legal_cards:
                reward = -10  # Pénalité pour action illégale
                return self.get_observation(), reward, False, {}

            # 🛑 Stocker l'état et l'action dans l'historique
            self.history.append((self.get_observation(), action))

            # Jouer la carte
            self.game.players[self.current_player].play_card(chosen_card)
            self.game.table.play(chosen_card, self.current_player)
            
            self.current_player = (self.current_player + 1) % 4
            if self.game.table.current_player == self.current_player:
                self.game.players[self.game.table.get_player_win()].get_team().add_round_score(self.game.table.get_points())
                self.current_player = self.game.table.get_player_win()
                self.game.table.plis_end()
            
            if self.game.round_finish():
                if not self.flag:
                    self.game.players[self.game.table.get_current_player()].get_team().add_round_score(10)
                    self.points_team[0] = self.game.players[0].get_team().get_round_score()
                    self.points_team[1] = self.game.players[1].get_team().get_round_score() 
                    _,_ = self.game.round_end()
                    self.game.table.empty_cards()
                    self.game.deck.cut()
                    self.game.deal_card(self.current_player, [3, 2, 3])
                    self.set_bid()
                    self.current_player = self.game.current_player
                    self.game.table.set_current_player(self.current_player)
                    self.flag = True
        
            # reward = self.compute_reward(self.current_player)
            # done = self.game.game_finish()
            
            return self.get_observation(), 0, False, {}
    
    def compute_reward(self,player):
        """Calcule une récompense pour l'IA en fonction de l'état du jeu."""
        team_score = self.game.players[player].get_team().get_round_score()  # Score de l'équipe de l'IA
        opponent_score = self.game.teams[1].get_round_score()  # Score de l'adversaire
        return team_score - opponent_score

    def render(self, mode="human"):
        """Affichage pour déboguer."""
        # print(f"État du jeu : {self.get_observation()}")

    def evaluate_card(self, card, coef = 3):
        """Donne un score à une carte selon sa puissance dans le jeu."""
        values = {"7": 0, "8": 0, "9": 0, "J": 2, "Q": 3, "K": 4,"10":10, "A": 11}
        values_asset = {"7": 0, "8": 0, "9": 14, "J": 20, "Q": 3, "K": 4,"10":10, "A": 11}

        if card.suit.name == self.game.table.get_current_asset():
            return values_asset[card.rank.value] / coef
        else:
            return values[card.rank.value] / coef
    
if __name__ == "__main__":
    env = CoincheEnv()
    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Action aléatoire pour tester
        obs, reward, done, _ = env.step(action)
        env.render()
