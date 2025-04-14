from Environnement.coinche import CoincheDeck, CoincheTable
from Environnement.player import Player
from Environnement.team import Team
from Environnement.utils import Rank


class CoinceGame:

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
    
    def __init__(self):
        self.deck = CoincheDeck()
        self.players = [Player(i) for i in range(4)]
        self.teams = [Team(0), Team(1)]  # Create two teams
        self.current_player = 0 # Index of current player
        self.table = CoincheTable(0)
        self.belote = False
        
        # Setup teams - players 0,2 are Team 0, players 1,3 are Team 1
        self.setup_teams()
        
        # Reference to the game in players for accessing opponents
        for player in self.players:
            player.game = self
    
    def setup_teams(self):
        # Assign players to teams (0,2 -> team 0, 1,3 -> team 1)
        self.teams[0].add_player(self.players[0])
        self.players[0].set_team(self.teams[0])
        self.teams[0].add_player(self.players[2])
        self.players[2].set_team(self.teams[0])
        self.teams[1].add_player(self.players[1])
        self.players[1].set_team(self.teams[1])
        self.teams[1].add_player(self.players[3])
        self.players[3].set_team(self.teams[1])

    def shuffle_deck(self):
        # Shuffle the deck
        self.deck.shuffle()


    def deal_card(self, player_id, deal_pattern):
        # Clear players' hands before dealing so that no player has extra cards
        for player in self.players:
            player.hand = []
    
        # No need to alter the actual self.players order for shuffling.
        # The deal_card method creates a temporary dealing order based on player_id,
        # preserving team assignments and overall player order.
        # Rearrange players so that dealing starts with the specified player_id
        players_order = self.players[player_id:] + self.players[:player_id]
        # Deal cards using the provided pattern
        self.deck.deal(players_order, deal_pattern)
        for player in self.players:
            self.sort_hand(player)
        
        self.current_player += 1
        if self.current_player >= 4:
            self.current_player = 0
    
    def sort_hand(self, player):
        player.hand.sort(key=lambda card: (card.suit.value, card.rank.value))

    def get_players_order(self, current_player):
        return self.players[current_player:] + self.players[:current_player]
    
    def get_table(self):
        return self.table
    
    def get_belote(self):
        return self.belote
    
    def set_belote(self,bool):
        self.belote = bool
    
    def get_legal_cards(self, player):

        # If the player is the first to play, they can play any card
        if self.table.get_first_card() is None:
            return player.hand
        
        elif self.table.get_current_asset() == "Full_ASSET":
            # If the player is not the first to play, they must follow suit
            first_card = self.table.get_first_card()
            best_card = self.table.get_best_card()
            play_hand = [card for card in player.hand if self.FULL_ASSET_POINTS[card.rank] > self.FULL_ASSET_POINTS[best_card.rank] and card.suit == first_card.suit]
            if play_hand == []:
                play_hand = [card for card in player.hand if card.suit == first_card.suit]
                if play_hand == []:
                    return player.hand
            return play_hand
        
        elif self.table.get_current_asset() == "No_ASSET":
            # If the player is not the first to play, they must follow suit
            first_card = self.table.get_first_card()
            play_hand = [card for card in player.hand if card.suit == first_card.suit]
            if play_hand == []:
                return player.hand
            return play_hand

        else:
            # If the player is not the first to play, they must follow suit
            first_card = self.table.get_first_card()
            best_card = self.table.get_best_card()
            asset_suit = self.table.get_current_asset()
            play_hand = [card for card in player.hand if card.suit == first_card.suit]
            if play_hand == []:
                if self.table.get_player_win() == (player.get_id()+2)%4:
                    return player.hand
                if best_card.suit.name == asset_suit:
                    play_hand = [card for card in player.hand if self.TRUMP_POINTS[card.rank] > self.TRUMP_POINTS[best_card.rank] and card.suit.name == asset_suit]
                    if play_hand != []:
                        return play_hand
                play_hand = [card for card in player.hand if card.suit.name == asset_suit]
                if play_hand == []:
                    return player.hand
            if first_card.suit.name == asset_suit:
                play_hand_asset = [card for card in player.hand if self.TRUMP_POINTS[card.rank] > self.TRUMP_POINTS[best_card.rank] and card.suit.name == asset_suit]
                if play_hand_asset != []:
                    return play_hand_asset
            return play_hand
        
    def get_legal_bids(self):
        bid = self.table.get_current_bid()
        if bid is None:
            score = 70
        else:
            score = bid.get_points()
        legal_bids = []
        for i in range(score+10, 170, 10):
            legal_bids.append(i)
        if score < 250:
            legal_bids.append(250)
        return legal_bids

    def round_finish(self):
        for player in self.players:
            if player.hand != []:
                return False
        return True
    
    def game_finish(self):
        for team in self.teams:
            if team.get_game_score() >= 2000:
                return True
        return False
    
    def announce_finish(self, player):
        # if self.table.current_bid is not None:
        #     print(self.table.current_bid.get_player() , player)
        if self.table.current_bid is not None and self.table.current_bid.get_player() == player:
            return True
        return False
    
    def round_end(self):
        point_bid = self.table.get_current_bid().get_points()
        team_bid = self.players[self.table.get_current_bid().get_player()].get_team().get_id()
        point_suit = self.table.get_current_bid().get_suit()
        coef_coinche = self.table.get_current_bid().get_coef()
        if point_suit:
            point_suit = point_suit.get_best_annonce()[1]
            team_suit = self.players[self.table.get_current_bid().get_suit().get_player()].get_team().get_id()
            if team_bid == team_suit:
                point_bid += point_suit
                point_suit = 0
        else:
            point_suit = 0
        if (self.teams[team_bid].get_round_score()+self.teams[team_bid].get_belote())  >= point_bid and self.teams[team_bid].get_round_score()  >= 82:
            self.teams[team_bid].add_game_score((self.teams[team_bid].get_round_score() + point_bid + self.teams[team_bid].get_belote())*coef_coinche)
            self.teams[(team_bid+1)%2].add_game_score((self.teams[(team_bid+1)%2].get_round_score() + point_suit + self.teams[(team_bid+1)%2].get_belote())*coef_coinche)
        else:
            self.teams[(team_bid+1)%2].add_game_score((160 + point_bid + point_suit + self.teams[(team_bid+1)%2].get_belote())*coef_coinche)
        self.belote = False
        for team in self.teams:
            team.set_belote(0)
        return team_bid,self.teams[team_bid].get_round_score()  >= point_bid
    
    def reset_round_score(self):
        for team in self.teams:
            team.reset_round_score()
            
                


        




   