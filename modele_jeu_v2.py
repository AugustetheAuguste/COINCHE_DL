from random import randint
import time
import gym
import numpy as np
from gym import spaces
from game import CoinceGame
from utils import Card, Rank, Suit
from stable_baselines3 import PPO

class CoincheEnv(gym.Env):

    NORMAL_POINTS = {
        Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
        Rank.TEN: 10, Rank.JACK: 2, Rank.QUEEN: 3,
        Rank.KING: 4, Rank.ACE: 11
    }
    
    TRUMP_POINTS = {
        Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 14,
        Rank.TEN: 10, Rank.JACK: 20, Rank.QUEEN: 3,
        Rank.KING: 4, Rank.ACE: 11
    }

    FULL_NORMAL_POINTS = {
        Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 0,
        Rank.TEN: 10, Rank.JACK: 2, Rank.QUEEN: 3,
        Rank.KING: 4, Rank.ACE: 19
    }

    FULL_ASSET_POINTS = {
        Rank.SEVEN: 0, Rank.EIGHT: 0, Rank.NINE: 9,
        Rank.TEN: 5, Rank.JACK: 14, Rank.QUEEN: 1,
        Rank.KING: 3, Rank.ACE: 6
    }

    def __init__(self, train_player=0):
        super(CoincheEnv, self).__init__()
        self.game = CoinceGame()
        
        # Charger l'ancien modèle
        self.old_model = PPO.load("mv2-4_1M_R")
        
        self.suit = ["SPADES", "HEARTS", "DIAMONDS", "CLUBS", "No_ASSET","Full_ASSET"]
        self.rank = ["7", "8", "9", "10", "J", "Q", "K", "A"]
        self.current_player = 0
        self.train_player = train_player  # Joueur contrôlé par le nouveau modèle
        self.order = []
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(48,), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.reset()

    def reset(self):
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
        self.game.table.announce((80, self.suit[couleur]), player)


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
    
    def step(self, action, training = True):
        obs = None
        player_ia = False
        reward = 0
        point = 0
        while not player_ia and not self.game.round_finish():
            legal_cards = self.game.get_legal_cards(self.game.players[self.current_player])

            if self.current_player == self.train_player:
                chosen_action = action  # Le joueur entraîné joue l'action choisie par PPO
                if chosen_action >= len(self.game.players[self.current_player].get_card()):
                    return self.get_observation(), -50, False, {}  
                chosen_card = self.game.players[self.current_player].get_card()[chosen_action]
                if chosen_card not in legal_cards:
                    return self.get_observation(), -50, False, {}
                if not training:
                    print(chosen_card)
                point = self.get_points(chosen_card)
            
            else:
                obs = self.get_observation()
                chosen_action, _ = self.old_model.predict(obs)  # Les autres joueurs utilisent l'ancien modèle
                if chosen_action < len(self.game.players[self.current_player].get_card()):
                    chosen_card = self.game.players[self.current_player].get_card()[chosen_action]
                    if chosen_card not in legal_cards:
                        chosen_card = legal_cards[randint(0, len(legal_cards)-1)]
                else:
                    chosen_card = legal_cards[randint(0, len(legal_cards)-1)]

            self.game.players[self.current_player].play_card(chosen_card)
            self.game.table.play(chosen_card, self.current_player)
            self.order.append(self.current_player)
            if self.current_player == self.train_player:
                obs = self.get_observation()
            self.current_player = (self.current_player + 1) % 4
            
            if len(self.game.table.get_card()) == 4:
                reward = self.game.table.get_points() 
                obs = self.get_observation()
                self.game.players[self.game.table.get_player_win()].get_team().add_round_score(reward)
                self.current_player = self.game.table.get_player_win()
                if not training:
                    print('ordre :', self.order[-4:])
                    print("carte joué : ",self.current_player,[str(card) for card in self.game.table.get_all_cards()[-4:]])
                    for player in self.game.players:
                        print([str(card) for card in player.get_card()])
                if self.current_player == (self.train_player+2)%4:
                    reward = point /2
                if self.current_player in [(self.train_player+1)%4, (self.train_player+3)%4]:
                    reward = -point
                self.game.table.plis_end()
                reward = reward*0.5

            if self.current_player == self.train_player:
                player_ia = True

        if self.game.round_finish():
            obs = self.get_observation()
            self.game.players[self.game.table.get_current_player()].get_team().add_round_score(10)
            reward = 2 * self.game.players[self.train_player].get_team().get_round_score() - self.game.players[(self.train_player+1)%2].get_team().get_round_score()
            _, _ = self.game.round_end()
            self.game.table.empty_cards()
            self.game.deck.cut()
            self.game.deal_card(self.current_player, [3, 2, 3])
            self.set_bid()
            self.current_player = self.game.current_player
            self.game.table.set_current_player(self.current_player)
            self.order = []

            
        done = self.game.game_finish()
        return obs, reward, done, {}

    def render(self, mode="human"):
        pass
        # print(f"État du jeu : {self.get_observation()}")

    def get_points(self, card):
        current_asset = self.game.table.get_current_asset()
        if current_asset == "Full_ASSET":
            return self.FULL_ASSET_POINTS[card.rank]
        elif current_asset == "No_ASSET":
            return self.FULL_NORMAL_POINTS[card.rank]
        elif card.suit.name == current_asset:
            return self.TRUMP_POINTS[card.rank]
        else:
            return self.NORMAL_POINTS[card.rank]
    