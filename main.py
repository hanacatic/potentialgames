import numpy as np
from game import Game, RandomIdenticalInterestGame 
from plot import *

RATIONALITY = 100
EPS = 1e-1

# payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # first identical interest game
# payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.1]]) # second identical interest game
# payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0.4, 0.4, 0.4, 0.5]]) # third identical interest game
# payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0.25 , 0.5]]) # fourth identical interest game
# payoff = np.array([[0, 0, 0, 0], [0.5, 1, 0, 0], [0, 0, 0, 0], [0, 0.25, 0, 0.5]]) # fifth identical interest game
# payoff = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 1, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.5]]) # sixth identical interest game
# payoff = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 1, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]) # seventh identical interest game
# payoff = np.array([[0.9, 0.9, 0.9, 0.9], [0.9, 1, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]]) # eighth identical interest game
# payoff = np.array([[0.9, 0.9, 0.9, 0.9], [0.9, 1, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.99]]) # ninth identical interest game
# payoff = np.array([[0.9, 0.9, 0.9, 0.9], [0.9, 1, 0.9, 0.9], [0.9, 0.95, 0.9, 0.9], [0.95, 0.95, 0.9, 0.99]]) # tenth identical interest game
# payoff = np.array([[0.9, 0.9, 0.9, 0.95], [0.9, 1, 0.9, 0.95], [0.9, 0.9, 0.9, 0.9], [0.95, 0.95, 0.9, 0.99]]) # eleventh identical interest game
payoff = np.array([[0, 0, 0.75, 0], [0.5, 1, 0, 0], [0.6, 0, 0, 0], [0, 0, 0, 0.5]]) # twelfth identical interest game
# payoff = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]) # thirteenth identical interest game
# payoff = np.array([[0.9, 0.9, 0.9, 0.9], [0.9, 1, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9]]) # fourteenth identical interest game
payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0.99, 0.9, 0.8, 0.5]]) # fifteenth identical interest game

# payoff = np.array([[0.9, 0, 0.25, 0], [0.5, 1, 0.75, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]]) # second identical interest game
# payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0, 0], [0, 0.9, 0.8, 0.5]]) # third identical interest game
# payoff = np.array([[0.5, 0, 0.25, 0], [0.9, 1, 0.75, 0], [0, 0.8, 0.5, 0.6], [0, 0.9, 0.8, 0.5]]) # fourth identical interest game
    
def mu(action_profile):
    return 1.0/16.0

def compute_beta(no_players, action_space, epsilon, delta):
    # return 1/max(epsilon/2, delta)*np.log(len(action_space)**no_players*(1-epsilon/2)*(4/(epsilon*len(action_space)**no_players*(epsilon/2)) - 1/(len(action_space)**no_players*(epsilon/2))))
    return 1/max(epsilon, delta)*np.log(len(action_space)**no_players/epsilon)

def compute_t(no_players, action_space, epsilon, delta):
    # return 25*no_players**2*len(action_space)**5*np.exp(4*beta)/16/np.pi**2*(np.log(np.log(len(action_space)**no_players)) + np.log(beta) + 2*np.log(4/epsilon))
    # return no_players**2*len(action_space)**5*(len(action_space)**no_players/epsilon)**(1/max(epsilon, delta))
    return no_players*(no_players**len(action_space)/epsilon)**(1/max(epsilon, delta))


# def main():
    # 
    # global payoff
    
    # # define a two player matrix game
    # no_players = 2
    # action_space = [0, 1, 2, 3]
    
    # save = False
    # show_plot = True
    # folder = 'Game 15'
    # plot_payoff(payoff, folder = folder, save = save)
    
    # delta = 0.01
            
    # for rationality in range( 0, RATIONALITY, 5): 
        
    #     game = Game(no_players, rationality, action_space, [utility_function_player_1, utility_function_player_2], potential_function, mu)
    #     game.set_initial_action_profile(np.array([3,3]))
    #     beta = compute_beta(no_players, action_space, EPS, delta)
    #     print('beta lower bound: ' + str(beta) + '\n')
    #     print('t lower bound: ' + str(compute_t(no_players, action_space, EPS, delta)) + '\n')
        
    #     N = 20
        
    #     potentials_history = np.zeros((N, game.max_iter))
    #     player_history = np.zeros((N, game.max_iter))
        
    #     for i in range(0, N):
            
    #         # game.sample_initial_action_profile(mu)

    #         game.play()
    #         potentials_history[i] = np.transpose(game.potentials_history).copy()
    #         player_history[i] = np.transpose(game.player_id_history).copy()
            
    #         # print(game.converged_iteration)
        
    #     print(rationality)
                
    #     mean_potential = np.mean(potentials_history, 0)
    #     std = np.std(potentials_history, 0)
    #     # plot_potential(mean_potential, title = 'mean_potential_N_' + str(N) +'_beta_' + str(rationality) + '_(' + str(game.initial_action_profile[0]) + '_' + str(game.initial_action_profile[1]) + ')', folder = folder, save = save)
    #     # plot_potential_with_std(mean_potential, std,  title = 'mean_potential_std_N_' + str(N) +'_beta_' + str(rationality) + '_(' + str(game.initial_action_profile[0]) + '_' + str(game.initial_action_profile[1]) + ')', folder = folder, save = save)
    #     # plot_history(potentials_history, title = 'potentials_history_N_' + str(N) +'_beta_' + str(rationality) + '_(' + str(game.initial_action_profile[0]) + '_' + str(game.initial_action_profile[1]) + ')', folder = folder, save = save)
    #     # plot_history(player_history, title = 'player_history_N_' + str(N) +'_beta_' + str(rationality) + '_(' + str(game.initial_action_profile[0]) + '_' + str(game.initial_action_profile[1]) + ')', folder = folder, save = save)

    #     plot_potential(mean_potential, title = 'mean_potential_N_' + str(N) +'_beta_' + str(rationality) + '_(sampled)', folder = folder, save = save)
    #     plot_potential_with_std(mean_potential, std,  title = 'mean_potential_std_N_' + str(N) +'_beta_' + str(rationality) + '_(sampled)', folder = folder, save = save)
    #     plot_history(potentials_history, title = 'potentials_history_N_' + str(N) +'_beta_' + str(rationality) + '_(sampled)', folder = folder, save = save)

    #     if show_plot:
    #         plt.show()
    

if __name__ == '__main__':
    
    # main()
    
    no_players = 2
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    game = RandomIdenticalInterestGame(action_space, firstNE, secondNE, 0.25)
    
    print(game.payoff)
    plot_payoff(game.payoff)
    plt.show(block = False)
    plt.pause(5)
    plt.close() 