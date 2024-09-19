import player

# constants 
RATIONALITY = 0 
EPS = 1e-3

class Game:
    self.max_iter = 1000
    
    def __init__(self, no_players, action_space, utility_functions, mu): # mu - initial distribution
        
        self.players = np.array([ Player(i, RATIONALITY, len(action_space), utility_functions[i]) for i in range(0, no_players)], dtype = object)
        
        self.action_profile = 
        
    def play(self):
        
        for i in range(0, self.max_iter):
            
            player_id = np.random.uniform(0, len(self.players), 1)
            
            self.action_profile[player_id] = self.players[player_id](self.action_profile)
    
