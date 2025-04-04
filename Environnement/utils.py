from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    SPADES = "♠"
    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"

    @classmethod
    def from_str(cls, valeur: str):
        """ Convertit un string (nom ou symbole) en Suit """
        for suit in cls:
            if valeur.upper() == suit.name or valeur == suit.value:
                return suit
        raise ValueError(f"Suit invalide : {valeur}")

class Rank(Enum):
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"

@dataclass
class Card:
    suit: Suit
    rank: Rank
    
    def __str__(self):
        return f"{self.rank.value}{self.suit.value}"