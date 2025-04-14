import numpy as np
import torch
from time import sleep
from Environnement.annonce_suit import Annonce
from Environnement.game import CoinceGame
from Environnement.utils import Card, Rank, Suit
from DQNannonce import DQNAgent
from stable_baselines3 import PPO

class MainCoincheAI:
    def __init__(self, state_size=68, action_size=60, training=True):
        """
        Initialise le jeu et l'agent DQN.
        """
        self.game = CoinceGame()
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)
        self.modele_jeu = Modele_jeu(self.game)
        self.list_annonce = {}
        self.suit_to_index = {"SPADES": 0.25, "HEARTS": 0.5, "DIAMONDS": 0.75, "CLUBS": 1.0 , "Full_ASSET": 1.25, "No_ASSET": 1.5}
        self.annonce_points = [80, 90, 100, 110, 120, 130, 140, 150, 160, 250]
        self.couleurs = ["HEARTS", "SPADES", "DIAMONDS", "CLUBS", "No_ASSET", "Full_ASSET"]
        self.bid =  [(val, couleur) for val in self.annonce_points for couleur in self.couleurs] + ["Passe"]


        self.training = training

    def start_game(self):
        self.list_annonce = {}
        self.modele_jeu.new_game(self.game)
        self.game.shuffle_deck()
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
        if not self.training:
            print(f"Annonce terminée ! Annonce finale : {self.game.table.get_current_bid().to_string()}")

        while not self.game.round_finish():
            player_order = self.game.get_players_order(self.game.table.current_player)
            for p in player_order:
                self.play(p)
            self.game.players[self.game.table.get_player_win()].get_team().add_round_score(self.game.table.get_points())
            self.game.table.set_current_player(self.game.table.get_player_win())
            self.game.table.plis_end()

        self.game.players[self.game.table.get_current_player()].get_team().add_round_score(10)
        if not self.training:
            print(f"Partie terminée ! Score de la manche : {self.game.teams[0].get_round_score()} - {self.game.teams[1].get_round_score()}")

        if self.training:
            for player in range(4):
                reward = self.calculate_reward(player, self.list_annonce[player][0][0])
                self.agent.remember(self.list_annonce[player][1], 
                                    self.list_annonce[player][4], 
                                    reward,
                                    self.list_annonce[player][2], 
                                    self.list_annonce[player][3])

        team, win = self.game.round_end()
        self.game.reset_round_score()
        if not self.training:
            print(f"L'équipe {team} a obtenu le résultat de {win}")
        self.game.table.empty_cards()
        self.modele_jeu.reset_order()

    def play(self, player):
        """L'agent DQN choisit une carte à jouer."""
        if isinstance(player, int):
            player_obj = self.game.players[player]
        else:
            player_obj = player

        legal_cards = self.game.get_legal_cards(player_obj)
        if not legal_cards:
            return

        valid = False
        while not valid:
            action = self.modele_jeu.play()
            if action < len(player_obj.get_card()):
                chosen_card = player_obj.get_card()[action]
                if chosen_card  in legal_cards:
                    valid = True

        self.modele_jeu.add_order(player_obj.get_id())
        player_obj.play_card(chosen_card)
        self.game.table.play(chosen_card, player_obj.get_id())

    def annonce(self, player):
        """L'agent DQN sélectionne automatiquement une annonce."""
        legal_bids = self.game.get_legal_bids()
        legal_actions = [(val, couleur) for val in legal_bids for couleur in self.couleurs] + ["Passe"]

        observation = self.get_observation(player)
        action = self.agent.act(observation)

        bid = self.bid[action]
        if bid not in legal_actions:
            self.agent.remember(observation, action, -10, self.get_observation(player), False)
            bid = "Passe"

        if bid != "Passe":
            # On utilise ici une couleur par défaut : "HEARTS"
            self.game.table.announce((bid[0], bid[1]), player)
            if not self.training:
                print(f"Le joueur {player} a annoncé {bid}")
                print([str(card) for card in self.game.players[player].get_card()])
        self.list_annonce[player] = (bid,observation,self.get_observation(player),self.game.announce_finish(player),action)

    def get_observation(self, player):
        observation = np.zeros(68)

        # 1️⃣ Cartes en main (52 bits, one-hot)
        if isinstance(player, int):
            player_obj = self.game.players[player]
        else:
            player_obj = player
            player = player_obj.get_id()

        cards = player_obj.get_card()
        for card in cards:
            index = self.card_to_index(card)
            observation[index] = 1
        
        for player in range(4):
            if player in self.list_annonce and self.list_annonce[player][0] != "Passe":
                observation[56 + player*2] = self.list_annonce[player][0][0] / 160.0 # Normalisé
                observation[56 + player*2 + 1] = self.suit_to_index.get(self.list_annonce[player][0][1], 0.0)
            else:
                observation[56 + player*2] = 0
                observation[56 + player*2 + 1] = 0


        # 3️⃣ Meilleure annonce actuelle
        current_bid = self.game.table.get_current_bid()
        if current_bid is not None:
            observation[64] = current_bid.get_points() / 160.0  # Normalisé
            observation[65] = self.suit_to_index.get(current_bid.get_trump_suit(), 0.0)
        else:
            observation[64] = 0
            observation[65] = 0

        # 4️⃣ Est-ce à mon tour ?
        observation[66] = 1 if self.game.table.current_player == player else 0

        # 5️⃣ Est-ce que mon équipe a la meilleure annonce actuelle ?
        if current_bid is not None:
            bidder_team = self.game.players[current_bid.get_player()].get_team()
            my_team = self.game.players[player].get_team()
            observation[67] = 1 if bidder_team == my_team else 0
        else:
            observation[67] = 0

        return observation


    def card_to_index(self, card):
        """Convertit une carte en indice unique (0 à 51)."""
        suits = {"HEARTS": 0, "DIAMONDS": 1, "CLUBS": 2, "SPADES": 3}
        ranks = {"SEVEN": 0, "EIGHT": 1, "NINE": 2, "TEN": 3, "JACK": 4, "QUEEN": 5, "KING": 6, "ACE": 7}
        return suits[card.suit.name] * 8 + ranks[card.rank.name]

    def calculate_reward(self, player, bid_value):
        if bid_value == "P":
            return 0  # Neutre si passe

        last_bid = self.game.table.get_current_bid()
        if last_bid.get_player() != player:
            return -0.2  # Petite pénalité pour annonce inutile

        team = self.game.players[player].get_team()
        team_score = team.get_round_score()
        contrat = last_bid.get_points()

        overbid = contrat - team_score

        # Contrat réussi
        if team_score >= contrat:
            # Plus c’est proche, mieux c’est (évite les gros contrats inutiles)
            return 5 - (overbid / 10)
        else:
            # Contrat raté : pénalité croissante avec l’écart
            return -2 - (abs(overbid) / 10)


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
    game_ai = MainCoincheAI()   

    episodes = 5000
    games_per_episode = 10  # Nombre de parties par épisode

    print("Phase de training")

    for episode in range(episodes):
        if episode % 100 == 0:
            print(f"Épisode {episode}/{episodes}")

        for game_count in range(games_per_episode):
            game_ai.start_game()

        # Entraîne le modèle avec l'expérience accumulée
        game_ai.train_agent()
        game_ai.agent.decay_epsilon()

    # Test après entraînement
    print("Phase de test (sans exploration)...")
    game_ai.training = False
    game_ai.start_game()

    # Sauvegarde du modèle
    game_ai.save_model("dqn_model.pth")

    # # Crée l'instance de l'IA avec le training désactivé
    # game_ai = MainCoincheAI(training=False)

    # # Charge le modèle DQN déjà entraîné
    # game_ai.agent.model.load_state_dict(torch.load("dqn_5K-10.pth", map_location=torch.device('cpu')))
    # game_ai.agent.model.eval()  # Mode évaluation pour éviter le dropout, etc.

    # # Lancer une partie avec affichage des annonces
    # print("Test du modèle DQN pour les annonces...")
    # game_ai.start_game()
