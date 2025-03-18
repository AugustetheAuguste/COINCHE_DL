from game import CoinceGame
from utils import Suit


class MainCoinche:
    def __init__(self):
        self.game = CoinceGame()
    
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
            team,win = self.game.round_end()
            print(f"l'equipe {team} a obtenu le resultat de {win}")
            self.game.table.empty_cards()
            print(f"Score total : {self.game.teams[0].get_game_score()} - {self.game.teams[1].get_game_score()}")
        print(f"Partie terminée ! Score final : {self.game.teams[0].get_game_score()} - {self.game.teams[1].get_game_score()}")
        if self.game.teams[0].get_game_score() > self.game.teams[1].get_game_score():
            print("L'équipe 1 a gagné !")
        else:    
            print("L'équipe 2 a gagné !")

    def play(self, player):
        legal_cards = self.game.get_legal_cards(player)

        # Affichage des cartes du joueur et des cartes légales
        print(f"Joueur {player.get_id()}:")
        print("Cartes en main :", [str(card) for card in player.get_card()])
        print("Cartes légales :", [str(card) for card in legal_cards])

        if not legal_cards:
            print("Aucune carte légale à jouer !")
            return  

        # Demande au joueur de choisir une carte
        while True:
            try:
                print("Choisissez une carte en entrant son numéro (0 pour la première, 1 pour la deuxième, etc.)")
                for i, card in enumerate(legal_cards):
                    print(f"{i}: {card}")

                choix = int(input("Votre choix : "))

                if 0 <= choix < len(legal_cards):
                    card = player.play_card(legal_cards[choix])  # Retire la carte de la main du joueur
                    self.game.table.play(card, player.get_id())  # Joue la carte sur la table
                    print(f"Le joueur {player} a joué {card}")
                    break
                else:
                    print("Choix invalide, veuillez entrer un nombre valide.")
            except ValueError:
                print("Entrée invalide, veuillez entrer un nombre.")

    def annonce(self, player):
        legal_bids = self.game.get_legal_bids()
        legal_bids.append("Passe")
        
        print(f"Joueur {self.game.players[player].get_id()}:")
        
        if self.game.table.get_current_bid() is not None:
            print(f"Dernière annonce : {self.game.table.get_current_bid().to_string()}")
        
        print("Cartes en main :", [str(card) for card in self.game.players[player].get_card()])
        
        # Étape 1 : Choisir la valeur de l'annonce
        print("Choisissez une valeur d'annonce :")
        
        for i, value in enumerate(legal_bids):
            print(f"{i}: {value}")

        while True:
            try:
                value_choice = int(input("Votre choix : "))
                if 0 <= value_choice < len(legal_bids):
                    bid_value = legal_bids[value_choice]
                    break
                else:
                    print("Choix invalide, veuillez entrer un nombre valide.")
            except ValueError:
                print("Entrée invalide, veuillez entrer un nombre.")

        # Filtrer les couleurs disponibles pour cette valeur d'annonce
        available_suits = [suit.name for suit in Suit]
        available_suits.append("Full_ASSET")  # Tout Atout
        available_suits.append("No_ASSET")  # Sans Atout
        if bid_value != "Passe":
            # Étape 2 : Choisir la couleur si plusieurs options sont disponibles
            print("Choisissez une couleur :")
            for i,suit in enumerate(available_suits):
                print(f"{i}: {suit}")

            while True:
                try:
                    suit_choice = int(input("Votre choix : "))
                    if 0 <= suit_choice < len(available_suits):
                        bid_suit = available_suits[suit_choice]
                        break
                    else:
                        print("Choix invalide, veuillez entrer un nombre valide.")
                except ValueError:
                    print("Entrée invalide, veuillez entrer un nombre.")

            # Enregistrer l'annonce
            self.game.table.announce((bid_value,bid_suit), player)
            print(f"Le joueur {player} a annoncé {(bid_value,bid_suit)}")



if __name__ == "__main__":
    MainCoinche().start_game()