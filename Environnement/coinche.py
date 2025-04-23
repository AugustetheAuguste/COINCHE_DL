import random
from typing import List, Optional, Tuple
from Environnement.player import Player
from Environnement.utils import Card, Rank, Suit


class CoincheDeck:
    def __init__(self):
        self.cards = [Card(suit, rank) 
                     for suit in Suit 
                     for rank in Rank]
        self.shuffle()
        
    def shuffle(self):
        random.shuffle(self.cards)

    def cut(self):
        cut = random.randint(5, len(self.cards)-5)
        self.cards = self.cards[cut:] + self.cards[:cut]
        
    def deal(self, players: List[Player], deal_pattern: Tuple[int, int, int]):
        card_index = 0
        for num_cards in deal_pattern:
            for player in players:
                for _ in range(num_cards):
                    if card_index < len(self.cards):
                        player.receive_card(self.cards[card_index])
                        card_index += 1

class CoincheBid:
    def __init__(self, points: int, trump_suit: Suit, player: Player):
        self.points = points
        self.trump_suit = trump_suit
        self.player = player
        self.is_coinched = False
        self.is_surcoinched = False
        self.suit = None
    
    def get_points(self):
        return self.points
    
    def get_trump_suit(self):
        return self.trump_suit
    
    def get_player(self):
        return self.player
    
    def get_coef(self):
        if self.is_coinched:
            if self.is_surcoinched:
                return 4
            return 2
        return 1
    
    def coinche(self):
        self.is_coinched = True

    def surcoinche(self):
        self.is_surcoinched = True

    def to_string(self):
        if self.trump_suit in ["Full_ASSET", "No_ASSET"]:
            return f"{self.points} points, atout {self.trump_suit}, joueur {self.player}"
        return f"{self.points} points, atout {self.trump_suit}, joueur {self.player}"
    
    def get_suit(self):
        return self.suit
    
    def set_suit(self, suit):
        self.suit = suit

class CoincheTable:
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

    def __init__(self,player):
        self.Card = []
        self.all_cards = []
        self.FirstCard = None
        self.best_card = None
        self.Player_win = None
        self.current_asset = None
        self.current_bid = None
        self.current_player = player

    def get_first_card(self):
        return self.FirstCard
    
    def get_player_win(self):
        return self.Player_win
    
    def get_card(self):
        return self.Card
    
    def get_all_cards(self):
        return self.all_cards
    
    def get_best_card(self):
        return self.best_card
    
    def set_best_card(self, card):
        self.best_card = card
    
    def add_card(self, card):
        self.Card.append(card)

    def set_first_card(self, card):
        self.FirstCard = card

    def plis_end(self):
        self.current_player = self.Player_win
        self.FirstCard = None
        self.best_card = None
        self.Card = []

    def empty_cards(self):
        self.current_asset = None
        self.current_bid = None
        self.all_cards = []

    def set_player_win(self, player):
        self.Player_win = player

    def set_current_asset(self, asset):
        self.current_asset = asset

    def get_current_asset(self):
        return self.current_asset
    
    def set_current_bid(self, bid):
        self.current_bid = bid

    def get_current_bid(self):
        return self.current_bid
    
    def get_current_player(self):
        return self.current_player
    
    def set_current_player(self, player):
        self.current_player = player
    
    def get_points(self):
        points = 0
        for card in self.Card:
            if self.current_asset == "Full_ASSET":
                points += self.FULL_ASSET_POINTS[card.rank]
            elif self.current_asset == "No_ASSET":
                points += self.FULL_NORMAL_POINTS[card.rank]
            elif card.suit.name == self.current_asset:
                points += self.TRUMP_POINTS[card.rank]
            else:
                points += self.NORMAL_POINTS[card.rank]
        return points
    
    def play(self, card, player):
        if self.FirstCard == None:
            self.set_first_card(card)
            self.set_best_card(card)
            self.set_player_win(player)
        elif self.current_asset == "Full_ASSET":
            if card.suit.name == self.FirstCard.suit.name:
                if self.FULL_ASSET_POINTS[card.rank] > self.FULL_ASSET_POINTS[self.best_card.rank]:
                    self.set_best_card(card)
                    self.set_player_win(player)
        elif self.current_asset == "No_ASSET":
            if card.suit.name == self.FirstCard.suit.name:
                if self.FULL_NORMAL_POINTS[card.rank] > self.FULL_NORMAL_POINTS[self.best_card.rank]:
                    self.set_best_card(card)
                    self.set_player_win(player)
        else:
            if card.suit.name == self.current_asset:
                if self.get_best_card().suit.name != self.current_asset:
                    self.set_best_card(card)
                    self.set_player_win(player)
                elif self.TRUMP_POINTS[card.rank] > self.TRUMP_POINTS[self.best_card.rank]:
                    self.set_best_card(card)
                    self.set_player_win(player)
            elif card.suit.name == self.get_best_card().suit.name:
                if self.NORMAL_POINTS[card.rank] > self.NORMAL_POINTS[self.best_card.rank]:
                    self.set_best_card(card)
                    self.set_player_win(player)
                
        self.add_card(card)
        self.all_cards.append(card)

    def announce(self, bid, player):
        self.set_current_bid(CoincheBid(bid[0], bid[1], player))
        self.set_current_asset(bid[1])


