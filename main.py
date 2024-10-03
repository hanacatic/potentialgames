import numpy as np
from game import Game, RandomIdenticalInterestGame 
from plot import *

import cProfile

RATIONALITY = 100
EPS = 0.5e-1
    
def mu(action_profile):
    return 1.0/16.0

def beta_experiments(delta = 0.25):
    
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    gameSetup = RandomIdenticalInterestGame(action_space, firstNE, secondNE, delta)
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(secondNE)
    beta_t = game.compute_beta(EPS)
    # game.set_max_iter(EPS)
    print(beta_t)
    print(game.max_iter)
    
    save = False 
    folder = 'WEEK 4'
    title = 'Average potential'
    n_exp = 25
    plot_payoff(game.gameSetup.payoff_player_1, folder = folder, save = save)
    mean_potential_history = np.zeros((6, game.max_iter))
    betas = np.arange(beta_t/2, beta_t + beta_t/8.0, beta_t/8.0)
    
    for idx, beta in enumerate(betas):
        print(beta)
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play(beta=beta)
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    mean_potential_history[5] = (1-EPS) * np.ones((1, game.max_iter))
    labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()
        
def delta_experiments(beta = 10, deltas = [0.9, 0.75, 0.5, 0.25, 0.1]):
    
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    save = False 
    folder = 'WEEK 4'
    title = 'Average potential'
    n_exp = 25    
    
    gameSetup = RandomIdenticalInterestGame(action_space, firstNE, secondNE, deltas[0])
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(secondNE)
    beta_t = game.compute_beta(EPS)
    plot_payoff(game.gameSetup.payoff_player_1)
    mean_potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
        
        print(delta)
        gameSetup.reset_payoff_matrix(delta)
        game.reset_game(gameSetup)
        plot_payoff(game.gameSetup.payoff_player_1)
        beta_t = game.compute_beta(EPS)
        print(beta_t)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play(beta=beta)
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    mean_potential_history[5] = (1-EPS) * np.ones((1, game.max_iter))
    labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title = title, folder = folder, save = save)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()

def epsilon_experiments(delta = 0.25, epsilons = [0.2, 0.1, 0.05, 0.01, 0.001]):
    
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    save = False 
    folder = 'WEEK 4'
    title = 'Average potential'
    n_exp = 25
            
    gameSetup = RandomIdenticalInterestGame(action_space, firstNE, secondNE, delta)
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(secondNE)
    plot_payoff(game.gameSetup.payoff_player_1)
    
    mean_potential_history = np.zeros((10, game.max_iter))
    
    for idx, eps in enumerate(epsilons):
        
        beta = game.compute_beta(eps)
        print(beta)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play(beta=beta)
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
        mean_potential_history[idx + 5] = (1 - eps) * np.ones((1, game.max_iter))
    labels = [ r'$\epsilon = 0.2$', r'$\epsilon = 0.1$', r'$\epsilon = 0.05$', r'$\epsilon = 0.01$', r'$\epsilon = 0.001$']
   
    plot_lines_eps_exp(mean_potential_history, labels, True, title = title, folder = folder, save = save)

    if not save:
        plt.show()
        # plt.show(block = False)
        # plt.pause(20)
        # plt.close()

def main():
    
    # beta_experiments(0.25)
    
    # delta_experiments()
    
    # epsilon_experiments()
    
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    save = False 
    folder = 'WEEK 4'
    title = 'Average potential'
    n_exp = 5
    
    # mean_potential_history = np.zeros((1, game.max_iter))
        
    gameSetup = RandomIdenticalInterestGame(action_space, firstNE, secondNE, 0.25)
    game = Game(gameSetup, algorithm = "log_linear_t", mu=mu)
    game.set_initial_action_profile(secondNE)
    plot_payoff(game.gameSetup.payoff_player_1)
    
    for _ in range(n_exp):
                
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play()
            
    mean_potential_history = np.mean(potentials_history, 0)
    
            
    print(game.action_profile)
    plot_potential(mean_potential_history)
    plt.show(block = False)
    plt.pause(20)
    plt.close()

if __name__ == '__main__':
    
    cProfile.run('main()')