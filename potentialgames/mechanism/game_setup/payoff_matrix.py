
class PayoffMatrix:
    
    def __init__(self, no_players, no_actions, firstNE, secondNE, delta, symmetric):
        
        self.no_players = no_players
        self.no_actions = no_actions
        self.firstNE = firstNE
        self.secondNE = secondNE
        self.delta = delta 
        self.symmetric = symmetric
        
    def generate_random_matrix(self, no_players=None, no_actions=None, firstNE=None, secondNE=None, delta=None, symmetric=None):
        
        if no_players is not None:
            self.no_players = no_players
        if no_actions is not None:
            self.no_actions = no_actions
        if firstNE is not None:
            self.firstNE = firstNE
        if secondNE is not None:
            self.secondNE = secondNE
        if delta is not None:
            self.delta = delta
        if symmetric is not None:
            self.symmetric = symmetric
            
        self.payoff_player_1 = np.random.uniform(0.0, 1 - self.delta, size = [self.no_actions] * self.no_players)
        
        self.payoff_player_1[tuple(self.firstNE)] = 1
        self.payoff_player_1[tuple(self.secondNE)] = 1 - self.delta
        
        if self.symmetric: 
            self.payoff_player_1 = make_symmetric_nd(self.payoff_player_1)
            
    def regenerate(self, method, **kwargs):
        """
        Regenerate the matrix, possibly with new properties.
        """
        method(self, **kwargs)