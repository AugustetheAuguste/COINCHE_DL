class Team:
    def __init__(self, team_id: int):
        self.team_id = team_id
        self.game_score = 0
        self.round_score = 0
        self.belote = 0
        self.players = []  # Will contain references to Player objects
    
    def add_player(self, player):
        self.players.append(player)
        player.team = self

    def set_belote(self , nbr):
        self.belote = nbr

    def get_belote(self):
        return self.belote

    def get_id(self):
        return self.team_id
    
    def get_players(self):
        return self.players
        
    def get_game_score(self):
        return self.game_score
    
    def get_round_score(self):
        return self.round_score
    
    def add_round_score(self, points: int):
        self.round_score += points

    def add_game_score(self, points: int):
        self.game_score += points

    def reset_round_score(self):
        self.round_score = 0

    def reset_game_score(self):
        self.game_score = 0
    
    def add_score(self, points: int):
        self.score += points

