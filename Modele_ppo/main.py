from Environnement.annonce_suit import Annonce
from Environnement.game import CoinceGame
from Environnement.utils import Card, Rank, Suit


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

            print("Annonce of card suit !")
            for _ in range(4):
                self.annonce_suit(player)
                player = (player + 1) % 4
            if self.game.table.get_current_bid().get_suit() is not None:
                print(f"Annonce de Suite terminée ! Annonce finale : {self.game.table.get_current_bid().get_suit().get_best_annonce()[2]} par le joueur {self.game.table.get_current_bid().get_suit().get_player()}")
            
            print(self.game.table.get_current_bid().get_trump_suit())
            player_order = self.game.get_players_order(self.game.table.current_player)
            if self.game.table.get_current_bid().get_suit():
                player_suit = self.game.table.get_current_bid().get_suit().get_player()
            else:
                player_suit = None 
            for player in player_order:
                if player_suit and player.get_id() == player_suit:
                    print(f"annonce du player {player_suit} : {self.game.table.get_current_bid().get_suit().get_best_annonce()[0]}")
                self.play(player)
            self.game.players[self.game.table.get_player_win()].get_team().add_round_score(self.game.table.get_points())
            self.current_player = self.game.table.get_player_win()
            self.game.table.plis_end()

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
                    if not self.game.get_belote():
                        player_hand = [str(card) for card in player.get_card()]
                        card = player.play_card(legal_cards[choix])  # Retire la carte de la main du joueur
                        if card.rank.name in ["QUEEN","KING"] and card.suit.name == self.game.table.get_current_bid().get_trump_suit():
                            if str(Card(Suit.from_str(self.game.table.get_current_bid().get_trump_suit()), Rank.QUEEN)) in player_hand and str(Card(Suit.from_str(self.game.table.get_current_bid().get_trump_suit()), Rank.KING)) in player_hand:
                                print(f"Belote Rebelote for player {player.get_id()}")
                                self.game.set_belote(True)
                                player.get_team().set_belote(20)
                    else:
                        card = player.play_card(legal_cards[choix])  # Retire la carte de la main du joueur
                    self.game.table.play(card, player.get_id())  # Joue la carte sur la table
                    print(f"Le joueur {player.get_id()} a joué {card}")
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

            choice = ["Coinche","Passe"]
            for i,choix in enumerate(choice):
                print(f"{i} : {choix}")
            while True:
                try:
                    choix = int(input("Votre choix : "))
                    if 0 <= choix < len(choice):
                        coinche = choice[choix]
                        print(coinche)
                        break
                    else:
                        print("Choix invalide, veuillez entrer un nombre valide.")
                except ValueError:
                    print("Entrée invalide, veuillez entrer un nombre.")
            if coinche == "Coinche":
                self.game.table.get_current_bid().coinche()
                choice = ["SurCoinche","Passe"]
                for i,choix in enumerate(choice):
                    print(f"{i} : {choix}")
                while True:
                    try:
                        choix = int(input("Votre choix : "))
                        if 0 <= choix < len(choice):
                            surcoinche = choice[choix]
                            print(surcoinche)
                            break
                        else:
                            print("Choix invalide, veuillez entrer un nombre valide.")
                    except ValueError:
                        print("Entrée invalide, veuillez entrer un nombre.")
                if surcoinche == "SurCoinche":
                    self.game.table.get_current_bid().surcoinche()

    def annonce_suit(self, player):
        annonce_suit = Annonce(self.game.players[player].get_card(), self.game.table.get_current_bid().get_trump_suit(), player)
        annonce = annonce_suit.get_annonces()

        if self.game.table.get_current_bid().get_suit() is not None:
            for i, value in enumerate(annonce):
                if value[1] < self.game.table.get_current_bid().get_suit().get_best_annonce()[1]:
                    annonce.pop(i)

        annonce.append("Passe")
        if not annonce:
            annonce = ["Passe"]

        if len(annonce) != 1:
            for i, value in enumerate(annonce):
                print(f"{i}: {value}")

            while True:
                try:
                    value_choice = int(input("Votre choix : "))
                    if 0 <= value_choice < len(annonce):
                        if annonce[value_choice] != "Passe":
                            if self.game.table.get_current_bid().get_suit() is not None :
                                if annonce[value_choice][1] == self.game.table.get_current_bid().get_suit().get_best_annonce()[1]:
                                    ordre = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
                                    if ordre.index(annonce[value_choice][3]) < ordre.index(self.game.table.get_current_bid().get_suit().get_best_annonce()[3]):
                                        break
                            self.game.table.get_current_bid().set_suit(annonce_suit)
                            self.game.table.get_current_bid().get_suit().set_best_annonce(annonce[value_choice])
                            print(f"Le joueur {player} a annoncé {annonce[value_choice][2]}")
                        break
                    else:
                        print("Choix invalide, veuillez entrer un nombre valide.")
                except ValueError:
                    print("Entrée invalide, veuillez entrer un nombre.")
        else:
            print(f"Le joueur {player} a annoncé {annonce[0]}")


if __name__ == "__main__":
    MainCoinche().start_game()