from Environnement.annonce_suit import Annonce
from Environnement.game import CoinceGame
from Environnement.utils import Card, Rank, Suit
import numpy as np
import torch
from DQNannonce import DQNAgent
from time import sleep

class MainCoincheAI:
    def __init__(self, state_size=52, action_size=10):
        """
        Initialise le jeu et l'agent DQN.
        """
        self.game = CoinceGame()
        self.agent = DQNAgent(state_size=state_size, action_size=action_size)

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
                self.game.table.set_current_player(self.game.table.get_player_win())
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
        """L'agent DQN choisit une carte à jouer."""
        # Si 'player' est un indice, récupérez l'objet Player, sinon utilisez-le directement.
        if isinstance(player, int):
            player_obj = self.game.players[player]
        else:
            player_obj = player

        legal_cards = self.game.get_legal_cards(player_obj)
        if not legal_cards:
            return

        observation = self.get_observation(player_obj)
        action = self.agent.act(observation)
        if action >= len(legal_cards):
            action = 0
        card = player_obj.play_card(legal_cards[action])
        self.game.table.play(card, player_obj.get_id())
        print(f"Le joueur {player_obj.get_id()} a joué {card}")

    def annonce(self, player, training=False):
        """L'agent DQN sélectionne automatiquement une annonce."""
        legal_bids = self.game.get_legal_bids() + ["Passe"]
        observation = self.get_observation(player)
        action = self.agent.act(observation)
        if action >= len(legal_bids):
            action = len(legal_bids) - 1
        bid_value = legal_bids[action]
        if bid_value != "Passe":
            # On utilise une couleur par défaut pour l'exemple
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
        Ici, player est l'indice du joueur.
        """
        if isinstance(player, int):
            player_obj = self.game.players[player]
        else:
            player_obj = player
        # Utilisez la méthode get_legal_cards de l'objet game en passant l'objet joueur.
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
            # Ici, vous pouvez ajouter du code pour récupérer et cumuler les récompenses ou scores de la partie
            total_reward += game_ai.game.teams[0].get_round_score() - game_ai.game.teams[1].get_round_score()
            sleep(2)

        # Mise à jour de l'agent sur l'ensemble des expériences collectées pendant cet épisode.
        game_ai.train_agent()
        game_ai.agent.decay_epsilon()
        print(f"Episode {episode + 1}/{episodes} terminé. Récompense totale de l'épisode : {total_reward}")
        sleep(1)

    game_ai.save_model("dqn_model.pth")