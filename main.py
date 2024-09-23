import numpy as np
from game import Game 
from plot import *

RATIONALITY = 0   
EPS = 1e-1

payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # trivial identical interest game
# payoff = np.array([[0.9, 0, 0.25, 0], [0.5, 1, 0.75, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # second identical interest game
# payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0, 0], [0, 0.9, 0.8, 0.5]]) # third identical interest game
# payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0, 0.9, 0.8, 0.5]]) # fourth identical interest game
# payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0.99, 0.9, 0.8, 0.5]]) # fifth identical interest game
    
def mu(action_profile):
    return 1.0/16.0

def utility_function_player_1(player_action, opponents_action):
    
    global payoff
    return payoff[player_action, opponents_action]

def utility_function_player_2(player_action, opponents_action):
    
    global payoff
    return np.transpose(payoff)[player_action, opponents_action]

def potential_function(action_profile):
    
    return payoff[action_profile[0], action_profile[1]]

def main():
    
    global payoff
    
    # define a two player matrix game
    no_players = 2
    action_space = [0, 1, 2, 3]
    game = Game(no_players, RATIONALITY, action_space, [utility_function_player_1, utility_function_player_2], potential_function, mu)
    
    print(game.initial_action_profile)
    
    N = 10 
    
    potentials_history = np.zeros((N, game.max_iter))
    
    for i in range(0, N):

        game.play()
        potentials_history[i] = np.transpose(game.potentials).copy()
    
    print(game.action_profile_history)
    
    plot_payoff(payoff)
    
    plot_potential(np.sum(potentials_history, 0)/N)
    
    plt.show()

if __name__ == '__main__':
    
    main()
