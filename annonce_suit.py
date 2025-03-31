from collections import Counter
from utils import Card, Rank, Suit

class Annonce:
    VALEURS_CARRES = {Rank.JACK: 200, Rank.NINE: 150, Rank.ACE: 100, Rank.TEN: 100, Rank.KING: 100, Rank.QUEEN: 100}
    VALEURS_SUITE = {5: 100, 4: 50, 3: 20}
    VALEUR_BELOTE = 20

    def __init__(self, main, atout, player):
        self.main = main  # Liste d'objets Card
        self.atout = atout
        self.annonces = self.detecter_annonces()
        self.best_annonce = max(self.annonces, key=lambda x: x[1], default=("Aucune", 0))
        self.player = player

    def detecter_annonces(self):
        annonces = []
        cartes_par_couleur = {suit: [] for suit in Suit}
        compte_valeurs = Counter()
        
        for card in self.main:
            cartes_par_couleur[card.suit].append(card.rank)
            compte_valeurs[card.rank] += 1
        
        # Détection des carrés
        for rank, count in compte_valeurs.items():
            if count == 4 and rank in self.VALEURS_CARRES:
                annonces.append((f"Carré de {rank.value}", self.VALEURS_CARRES[rank], "carré", rank.value))
        
        # Détection des suites (Cent, Cinquante, Tierce)
        ordre = [Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]
        for suit, ranks in cartes_par_couleur.items():
            indices = sorted([ordre.index(rank) for rank in ranks if rank in ordre])
            longest_suite = self.trouver_suite(indices)
            if longest_suite in self.VALEURS_SUITE:
                annonces.append((f"{[ordre[i].value for i in indices[:longest_suite]]} dans {suit.value}", self.VALEURS_SUITE[longest_suite], f"suite de {longest_suite}", ordre[indices[:longest_suite][-1]].value))
        
        # Détection de la Belote
        if Card(self.atout, Rank.KING) in self.main and Card(self.atout, Rank.QUEEN) in self.main:
            annonces.append(("Belote", self.VALEUR_BELOTE, "belote", "KQ"))
        
        return annonces
    
    def trouver_suite(self, indices):
        if not indices:
            return 0
        
        max_longueur = 1
        longueur_actuelle = 1
        for i in range(1, len(indices)):
            if indices[i] == indices[i-1] + 1:
                longueur_actuelle += 1
            else:
                max_longueur = max(max_longueur, longueur_actuelle)
                longueur_actuelle = 1
        return max(max_longueur, longueur_actuelle)
    
    def get_annonces(self):
        return self.annonces
    
    def get_best_annonce(self):
        return self.best_annonce
    
    def set_best_annonce(self, annonce):
        self.best_annonce = annonce

    def get_player(self):
        return self.player

    def total_points(self):
        return sum(points for _, points in self.annonces)

    def __str__(self):
        return "\n".join(["{}: {} points".format(desc, points) for desc, points in self.annonces])