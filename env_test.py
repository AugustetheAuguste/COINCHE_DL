from random import randint
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from game import CoinceGame
from utils import Card, Rank, Suit

class CoincheEnv(gym.Env):
    def __init__(self,model):
        super(CoincheEnv, self).__init__()
        self.game = CoinceGame()
        self.modele = PPO.load(model)

        self.suit = ["SPADES", "HEARTS", "DIAMONDS", "CLUBS", "No_ASSET","Full_ASSET"]
        self.rank = ["7", "8", "9", "10", "J", "Q", "K", "A"]
        self.current_player = 0
        self.order = []

        # Définition des espaces d'observation et d'action
        self.observation_space = spaces.Box(low=0, high=1, shape=(48,), dtype=np.float32)  # Représentation des cartes en main
        self.action_space = spaces.Discrete(8)  # Nombre total d'actions possibles (annonces, jeux de cartes, etc.)

        self.reset()

    def reset(self):
        """Réinitialise le jeu et retourne l'état initial."""
        self.game = CoinceGame()
        self.order = []
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
                observation[i] = self.card_to_index(cards_in_hand[i]) / 52.0 # Normalisé
            else:
                observation[i] = -1  # Placeholder pour cartes manquantes

        # 2️⃣ Atout (1 entrée, normalisé)
        observation[8] = self.suit.index(self.game.table.get_current_asset()) / 6.0

        # 3️⃣ Contrat (1 entrée, normalisé)
        observation[9] = ( self.game.table.get_current_bid().get_points() - 80 ) / 80

        # 4️⃣ Coinche (1 entrée)
        observation[10] = self.game.table.get_current_bid().get_coef() / 4.0

        # 5️⃣ Tour actuel (1 entrée, normalisé)
        observation[11] = (8 - len(self.game.players[0].get_card())) / 8.0

        # 6️⃣ Cartes posées (4 entrées, index normalisés ou -1 si vide)
        observation[12:16] = -1  # Initialiser à -1
        last_plis = self.game.table.get_card()
        for i, card in enumerate(last_plis):
            index = 12 + self.order[i-len(last_plis)]
            observation[index] = self.card_to_index(card) / 52.0

        # 7️⃣ Plis précédents (32 entrées max, 7 derniers plis)
        observation[16:48]
        old_plis = self.game.table.get_all_cards()  # On prend les 7 derniers plis max
        for i, card in enumerate(old_plis):
            index = 16 + 8*self.order[i] + i//4
            observation[index] = self.card_to_index(card) / 52.0

        return observation

    def card_to_index(self, card):
        """Convertit une carte en un index unique dans l'observation."""
        suits = {"HEARTS": 0, "DIAMONDS": 1, "CLUBS": 2, "SPADES": 3}
        ranks = {"7": 0, "8": 1, "9": 2, "10": 3, "J": 4, "Q": 5, "K": 6, "A": 7}
        return suits[card.suit.name] * 8 + ranks[card.rank.value]

    def round(self, info=False):
        
        if info:
            for player in env.game.players:
                print([str(card) for card in player.get_card()])
            print(env.game.table.get_current_bid().get_trump_suit())
            print(env.game.table.get_current_bid().get_player())

        while not self.game.round_finish():
            legal_cards = self.game.get_legal_cards(self.game.players[self.current_player])
            if len(legal_cards) == 1:
                chosen_card = legal_cards[0]
            while True:
                action,_ = self.modele.predict(self.get_observation())
                if action < len(self.game.players[self.current_player].get_card()):
                    chosen_card = self.game.players[self.current_player].get_card()[action]
                    if chosen_card in legal_cards:
                        break

            # Jouer la carte
            if info:
                print("carte joué : ",self.current_player,str(chosen_card))
            self.game.players[self.current_player].play_card(chosen_card)
            self.game.table.play(chosen_card, self.current_player)
            self.order.append(self.current_player)
            
            self.current_player = (self.current_player + 1) % 4
            if self.game.table.current_player == self.current_player:
                self.game.players[self.game.table.get_player_win()].get_team().add_round_score(self.game.table.get_points())
                self.current_player = self.game.table.get_player_win()
                if info : 
                    print("Player {} wins the trick".format(self.current_player))
                    for player in self.game.players:
                            print([str(card) for card in player.get_card()])
                self.game.table.plis_end()
        
        self.game.players[self.game.table.get_current_player()].get_team().add_round_score(10)
        point_team_0 = self.game.players[0].get_team().get_round_score()
        point_team_1 = self.game.players[1].get_team().get_round_score() 
        self.order = []
        _,_ = self.game.round_end()
        self.game.table.empty_cards()
        self.game.deck.cut()
        self.game.deal_card(self.current_player, [3, 2, 3])
        self.set_bid()
        self.current_player = self.game.current_player
        self.game.table.set_current_player(self.current_player)
        self.flag = True
        
        return point_team_0, point_team_1

if __name__ == "__main__":
    env = CoincheEnv("mv2-4_1M_R")
    p0,p1 = env.round(True)
    print("score team 0 : ",p0)
    print("score team 1 : ",p1)