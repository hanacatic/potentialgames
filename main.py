import numpy as np
import matplotlib.pyplot as plt

from game import Game 

RATIONALITY = 50   
EPS = 1e-1

# # payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # trivial identical interest game
# # payoff = np.array([[0.9, 0, 0.25, 0], [0.5, 1, 0.75, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # second identical interest game
# payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0, 0], [0, 0.9, 0.8, 0.5]]) # third identical interest game
# # payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0, 0.9, 0.8, 0.5]]) # fourth identical interest game



def mu(action_profile):
    return 1.0/16.0

# def utility_function_player_1(player_action, opponents_action):
    
#     global payoff
#     # payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # trivial identical interest game
#     # payoff = np.array([[0.9, 0, 0.25, 0], [0.5, 1, 0.75, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # second identical interest game
#     # payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0, 0], [0, 0.9, 0.8, 0.5]]) # third identical interest game
#     # payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0, 0.9, 0.8, 0.5]]) # fourth identical interest game

#     # payoff = np.zeros((4, 4))
#     # payoff[1,1] = 1 

#     return payoff[player_action, opponents_action]

# def utility_function_player_2(player_action, opponents_action):
    
#     global payoff

#     # payoff = np.zeros((4, 4))
#     # payoff[1,1] = 1 

#     return np.transpose(payoff)[player_action, opponents_action]

def utility_function(player_id, player_action, opponents_action):
    
    # global payoff
    # payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # trivial identical interest game
    # payoff = np.array([[0.9, 0, 0.25, 0], [0.5, 1, 0.75, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # second identical interest game
    # payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0, 0], [0, 0.9, 0.8, 0.5]]) # third identical interest game
    payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0, 0.9, 0.8, 0.5]]) # fourth identical interest game

    # payoff = np.zeros((4, 4))
    # payoff[1,1] = 1 
    if player_id == 0:
        return payoff[player_action, opponents_action]
    return np.transpose(payoff)[player_action, opponents_action]

if __name__ == '__main__':
    
    no_players = 2

    action_space = [0, 1, 2, 3]

    game = Game(no_players, RATIONALITY, action_space, [utility_function, utility_function], utility_function, mu)
    
    print(game.action_profile)
 
    game.play()
    print(game.action_profile)
    
    plt.plot(game.potentials)
    plt.ylabel('Potential')
    plt.xlabel('Iteration')
    plt.title('Second game ( beta = ' + str(RATIONALITY) + ', mu = uniform, initial action profile: ' + str(game.initial_action_profile) + ' )')
    plt.show()
