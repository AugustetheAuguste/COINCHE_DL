from annonce_suit import Annonce
from game import CoinceGame
from utils import Card, Rank, Suit
from stable_baselines3 import PPO
import numpy as np

class MainCoincheAI:
    def __init__(self, model_path="coinche_ia"):
        self.game = CoinceGame()
        self.model = PPO.load(model_path)

    def start_game(self):
        self.game.shuffle_deck()

        while not self.game.game_finish():
            self.game.deck.cut()
            self.game.deal_card(self.game.current_player, [3, 2, 3])
            player = self.game.current_player
            self.game.table.set_current_player(player)

            while not self.game.announce_finish(player):
                self.annonce(player)
                player = (player + 1) % 4
                if player == self.game.current_player and self.game.table.get_current_bid() is None:
                    self.game.deck.cut()
                    self.game.deal_card(self.game.current_player, [3, 2, 3])
                    player = self.game.current_player
            print(f"Annonce terminée ! Annonce finale : {self.game.table.get_current_bid().to_string()}")

            print("Annonce of card suit !")
            for _ in range(4):
                self.annonce_suit(player)
                player = (player + 1) % 4

            if self.game.table.get_current_bid().get_suit() is not None:
                print(f"Annonce de Suite terminée ! Annonce finale : {self.game.table.get_current_bid().get_suit().get_best_annonce()[2]} par le joueur {self.game.table.get_current_bid().get_suit().get_player()}")

            print(self.game.table.get_current_bid().get_trump_suit())
            player_order = self.game.get_players_order(self.game.table.current_player)
            for player in player_order:
                
                self.play(player)

            while not self.game.round_finish():
                player_order = self.game.get_players_order(self.game.table.current_player)
                for player in player_order:
                    self.play(player)

                self.game.players[self.game.table.get_player_win()].get_team().add_round_score(self.game.table.get_points())
                self.current_player = self.game.table.get_player_win()
                self.game.table.plis_end()
                print(f"Manche terminée ! Manche remportée par le joueur {self.game.table.get_player_win()}")

            self.game.players[self.game.table.get_current_player()].get_team().add_round_score(10)
            print(f"Partie terminée ! Score de la manche : {self.game.teams[0].get_round_score()} - {self.game.teams[1].get_round_score()}")
            team, win = self.game.round_end()
            print(f"L'équipe {team} a obtenu le résultat de {win}")
            self.game.table.empty_cards()
            print(f"Score total : {self.game.teams[0].get_game_score()} - {self.game.teams[1].get_game_score()}")

        print(f"Partie terminée ! Score final : {self.game.teams[0].get_game_score()} - {self.game.teams[1].get_game_score()}")
        if self.game.teams[0].get_game_score() > self.game.teams[1].get_game_score():
            print("L'équipe 1 a gagné !")
        else:
            print("L'équipe 2 a gagné !")

    def play(self, player):
        """Sélectionne automatiquement une carte à jouer en utilisant l'IA."""
        legal_cards = self.game.get_legal_cards(player)
        if not legal_cards:
            return

        observation = self.get_observation(player)
        action, _ = self.model.predict(observation)
        if action >= len(legal_cards):
            action = 0  # Sélection d'une carte valide par défaut

        card = player.play_card(legal_cards[action])

        #
        if not self.game.get_belote():
            player_hand = [str(card) for card in player.get_card()]
            card = player.play_card(legal_cards[action])  # Retire la carte de la main du joueur
            if card.rank.name in ["QUEEN","KING"] and card.suit.name == self.game.table.get_current_bid().get_trump_suit():
                if str(Card(Suit.from_str(self.game.table.get_current_bid().get_trump_suit()), Rank.QUEEN)) in player_hand and str(Card(Suit.from_str(self.game.table.get_current_bid().get_trump_suit()), Rank.KING)) in player_hand:
                    self.game.set_belote(True)
                    player.get_team().set_belote(20)
        else:
            card = player.play_card(legal_cards[action]) 
        #

        self.game.table.play(card, player.get_id())
        print(f"Le joueur {player.get_id()} a joué {card}")

    def annonce(self, player):
        """L'IA sélectionne automatiquement une annonce."""
        legal_bids = self.game.get_legal_bids() + ["Passe"]
        observation = self.get_observation(player)
        action, _ = self.model.predict(observation)

        if action >= len(legal_bids):
            action = len(legal_bids) - 1  # Choix par défaut : "Passe"

        bid_value = legal_bids[action]
        if bid_value != "Passe":
            self.game.table.announce((bid_value, "HEARTS"), player)  # Couleur par défaut
            print(f"Le joueur {player} a annoncé {bid_value}")

    def annonce_suit(self, player):
        """L'IA sélectionne automatiquement une annonce de suite."""
        annonce_suit = Annonce(self.game.players[player].get_card(), self.game.table.get_current_bid().get_trump_suit(), player)
        annonce = annonce_suit.get_annonces()

        if self.game.table.get_current_bid().get_suit() is not None:
            annonce = [a for a in annonce if a[1] >= self.game.table.get_current_bid().get_suit().get_best_annonce()[1]]

        annonce.append("Passe")
        observation = self.get_observation(player)
        action, _ = self.model.predict(observation)

        if action >= len(annonce):
            action = len(annonce) - 1  # Choix par défaut : "Passe"

        if annonce[action] != "Passe":
            self.game.table.get_current_bid().set_suit(annonce_suit)
            self.game.table.get_current_bid().get_suit().set_best_annonce(annonce[action])
            print(f"Le joueur {player} a annoncé {annonce[action][2]}")

    def get_observation(self, player):
        """Retourne l'observation de l'état du jeu sous forme de tableau normalisé."""
        observation = np.zeros(52)
        for card in player.get_card():
            index = self.card_to_index(card)
            observation[index] = 1
        return observation

    def card_to_index(self, card):
        """Convertit une carte en un index unique dans l'observation."""
        suits = {"HEARTS": 0, "DIAMONDS": 1, "CLUBS": 2, "SPADES": 3}
        ranks = {"7": 0, "8": 1, "9": 2, "10": 3, "J": 4, "Q": 5, "K": 6, "A": 7}
        return suits[card.suit.name] * 8 + ranks[card.rank.name]


if __name__ == "__main__":
    MainCoincheAI().start_game()
