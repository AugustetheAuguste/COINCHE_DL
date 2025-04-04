from typing import List

from Environnement.utils import Card

class Player:
    def __init__(self, position: int):
        self.position = position
        self.hand: List[Card] = []
        self.team = None  # Will be set when added to a team
        
    def receive_card(self, card: Card):
        self.hand.append(card)
        
    def play_card(self, card: Card):
        self.hand.remove(card)
        return card
    
    def get_card(self):
        return self.hand
    
    def get_team(self):
        return self.team
    
    def set_team(self, team):
        self.team = team
    
    def get_id(self):
        return self.position
    
    def get_teammate(self):
        """Returns the teammate of this player"""
        if self.team:
            return next((p for p in self.team.players if p != self), None)
        return None
    
    def get_opponents(self):
        """Returns a list of opponent players"""
        opponents = []
        for team in self.team.game.teams:
            if team != self.team:
                opponents.extend(team.players)
        return opponents

