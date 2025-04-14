import numpy as np
import torch
from time import sleep
from Environnement.annonce_suit import Annonce
from Environnement.game import CoinceGame
from Environnement.utils import Card, Rank, Suit
from DQNannonce import DQNAgent
from stable_baselines3 import PPO

class MainCoincheAI:
    def __init__(self, state_size=52, action_size=10):
        """
        Initialise le jeu et l'agent DQN.
        """
        self.game = CoinceGame()
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)
        self.modele_jeu = Modele_jeu(self.game)

    def start_game(self):
        self.game = CoinceGame()
        self.modele_jeu.new_game(self.game)
        self.game.shuffle_deck()
        # Boucle principale du jeu : on simule une partie complète
        while not self.game.game_finish():
            self.game.deck.cut()
            self.game.deal_card(self.game.current_player, [3, 2, 3])
            player = self.game.current_player
            self.game.table.set_current_player(player)

            # Phase d'annonce
            while not self.game.announce_finish(player):
                self.annonce(player)
                player = (player + 1) % 4
                if player == self.game.current_player and self.game.table.get_current_bid() is None:
                    self.game.deck.cut()
                    self.game.deal_card(self.game.current_player, [3, 2, 3])
                    player = self.game.current_player
            print(f"Annonce terminée ! Annonce finale : {self.game.table.get_current_bid().to_string()}")

            # Phase d'annonce de suite
            print("Annonce de couleur (suite) !")
            for _ in range(4):
                self.annonce_suit(player)
                player = (player + 1) % 4

            if self.game.table.get_current_bid().get_suit() is not None:
                print(f"Annonce de Suite terminée ! Annonce finale : {self.game.table.get_current_bid().get_suit().get_best_annonce()[2]} par le joueur {self.game.table.get_current_bid().get_suit().get_player()}")

            print(f"Atout pour l'annonce : {self.game.table.get_current_bid().get_trump_suit()}")
            # Phase de jeu
            player_order = self.game.get_players_order(self.game.table.current_player)
            for p in player_order:
                self.play(p)

            while not self.game.round_finish():
                player_order = self.game.get_players_order(self.game.table.current_player)
                for p in player_order:
                    self.play(p)
                self.game.players[self.game.table.get_player_win()].get_team().add_round_score(self.game.table.get_points())
                self.game.table.set_current_player(self.game.table.get_player_win())
                self.game.table.plis_end()
                print(f"Manche terminée ! Manche remportée par le joueur {self.game.table.get_player_win()}")

            self.game.players[self.game.table.get_current_player()].get_team().add_round_score(10)
            print(f"Partie terminée ! Score de la manche : {self.game.teams[0].get_round_score()} - {self.game.teams[1].get_round_score()}")
            team, win = self.game.round_end()
            self.game.reset_round_score()
            print(f"L'équipe {team} a obtenu le résultat de {win}")
            self.game.table.empty_cards()
            self.modele_jeu.reset_order()
            print(f"Score total : {self.game.teams[0].get_game_score()} - {self.game.teams[1].get_game_score()}")

        print(f"Partie terminée ! Score final : {self.game.teams[0].get_game_score()} - {self.game.teams[1].get_game_score()}")
        if self.game.teams[0].get_game_score() > self.game.teams[1].get_game_score():
            print("L'équipe 1 a gagné !")
        else:
            print("L'équipe 2 a gagné !")

    def play(self, player):
        """L'agent DQN choisit une carte à jouer."""
        if isinstance(player, int):
            player_obj = self.game.players[player]
        else:
            player_obj = player

        legal_cards = self.game.get_legal_cards(player_obj)
        if not legal_cards:
            return

        # observation = self.get_observation(player_obj)
        # action = self.agent.act(observation)
        valid = False
        while not valid:
            action = self.modele_jeu.play()
            if action < len(player_obj.get_card()):
                chosen_card = player_obj.get_card()[action]
                if chosen_card  in legal_cards:
                    valid = True

        # if action >= len(legal_cards):
        #     action = 0
        self.modele_jeu.add_order(player_obj.get_id())
        player_obj.play_card(chosen_card)
        self.game.table.play(chosen_card, player_obj.get_id())
        print(f"Le joueur {player_obj.get_id()} a joué {chosen_card}")

    def annonce(self, player, training=False):
        """L'agent DQN sélectionne automatiquement une annonce."""
        legal_bids = self.game.get_legal_bids() + ["Passe"]
        observation = self.get_observation(player)
        action = self.agent.act(observation)
        if action >= len(legal_bids):
            action = len(legal_bids) - 1
        bid_value = legal_bids[action]
        if bid_value != "Passe":
            # On utilise ici une couleur par défaut : "HEARTS"
            self.game.table.announce((bid_value, "HEARTS"), player)
            print(f"Le joueur {player} a annoncé {bid_value}")

        if training:
            next_obs = self.get_observation(player)
            reward = self.calculate_reward(player, bid_value)
            done = self.game.announce_finish(player)
            self.agent.remember(observation, action, reward, next_obs, done)

    def annonce_suit(self, player):
        """L'agent DQN sélectionne automatiquement une annonce de suite."""
        annonce_suit = Annonce(self.game.players[player].get_card(),
                               self.game.table.get_current_bid().get_trump_suit(),
                               player)
        annonce = annonce_suit.get_annonces()
        if self.game.table.get_current_bid().get_suit() is not None:
            annonce = [a for a in annonce if a[1] >= self.game.table.get_current_bid().get_suit().get_best_annonce()[1]]
        annonce.append("Passe")
        observation = self.get_observation(player)
        action = self.agent.act(observation)
        if action >= len(annonce):
            action = len(annonce) - 1
        if annonce[action] != "Passe":
            self.game.table.get_current_bid().set_suit(annonce_suit)
            self.game.table.get_current_bid().get_suit().set_best_annonce(annonce[action])
            print(f"Le joueur {player} a annoncé {annonce[action][2]}")

    def get_observation(self, player):
        """
        Retourne l'état du joueur sous forme de vecteur (taille 52).
        """
        if isinstance(player, int):
            player_obj = self.game.players[player]
        else:
            player_obj = player
        legal_cards = self.game.get_legal_cards(player_obj)
        observation = np.zeros(52)
        for card in legal_cards:
            index = self.card_to_index(card)
            observation[index] = 1
        return observation

    def card_to_index(self, card):
        """Convertit une carte en indice unique (0 à 51)."""
        suits = {"HEARTS": 0, "DIAMONDS": 1, "CLUBS": 2, "SPADES": 3}
        ranks = {"SEVEN": 0, "EIGHT": 1, "NINE": 2, "TEN": 3, "JACK": 4, "QUEEN": 5, "KING": 6, "ACE": 7}
        return suits[card.suit.name] * 8 + ranks[card.rank.name]

    def calculate_reward(self, player, bid_value):
        """Calcule une récompense simple pour l'annonce."""
        if bid_value == "Passe":
            return 0
        elif self.game.table.get_current_bid() is None or bid_value > self.game.table.get_current_bid().value:
            return 1
        else:
            return -1

    def save_model(self, filepath="dqn_model.pth"):
        """Sauvegarde le modèle DQN."""
        torch.save(self.agent.model.state_dict(), filepath)
        print(f"Modèle sauvegardé dans {filepath}")

    def train_agent(self, batch_size=32):
        """Entraîne l'agent DQN sur les expériences sauvegardées."""
        self.agent.replay(batch_size)
        self.agent.update_target_model()

class Modele_jeu:

    def __init__(self, game):
        self.game = game
        self.order = []
        self.modele = PPO.load("mv2-5_5M_R")
        self.suit = ["SPADES", "HEARTS", "DIAMONDS", "CLUBS", "No_ASSET","Full_ASSET"]
        self.rank = ["7", "8", "9", "10", "J", "Q", "K", "A"]

    def add_order(self,player):
        self.order.append(player)

    def new_game(self,game):
        self.game = game
        self.order = []

    def reset_order(self):
        self.order = []

    def get_observation(self):
        """Retourne une observation complète en encodant mieux la main du joueur."""
        
        observation = np.zeros(48)  # 8 (main) + 1 (atout) + 1 (contrat) + 1 (coinche) + 1 (tour) + 4 (cartes posées) + 28 (plis précédents)

        current_player = self.game.table.current_player

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
    
    def play(self):
        obs = self.get_observation()
        action,_ = self.modele.predict(obs)
        return action
    
if __name__ == "__main__":
    state_size = 52
    action_size = 10
    game_ai = MainCoincheAI(state_size=state_size, action_size=action_size)

    episodes = 5
    games_per_episode = 3  # Nombre de parties par épisode
    for episode in range(episodes):
        total_reward = 0
        for game_count in range(games_per_episode):
            game_ai.game.shuffle_deck()
            sleep(1)
            game_ai.start_game()  # Démarre une partie complète
            total_reward += game_ai.game.teams[0].get_round_score() - game_ai.game.teams[1].get_round_score()
            sleep(2)
        game_ai.train_agent()
        game_ai.agent.decay_epsilon()
        print(f"Episode {episode+1}/{episodes} terminé. Récompense totale : {total_reward}")
        sleep(1)
    game_ai.save_model("dqn_model.pth")