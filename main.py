import numpy as np
from game import Game, RandomIdenticalInterestGame 
from plot import *

RATIONALITY = 100
EPS = 1e-1
    
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
    folder = 'WEEK 3'
    title = 'Average potential'
    n_exp = 25
    plot_payoff(game.game.payoff, folder = folder, save = save)
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
        
def delta_experiments():
    
    action_space = [0, 1, 2, 3]
    
    firstNE = np.array([1,1])
    secondNE = np.array([3,3])
    
    delta = 0.25
    
    gameSetup = RandomIdenticalInterestGame(action_space, firstNE, secondNE, delta)
    game = Game(gameSetup, mu=mu)
    game.set_initial_action_profile(secondNE)
    beta_t = game.compute_beta(EPS)
    # game.set_max_iter(EPS)
    print(beta_t)
    print(game.max_iter)
    plt.show(block = False)
    plt.pause(3)
    plt.close()
               
def main():
    
    beta_experiments(0.25)
    
    delta_experiments()

if __name__ == '__main__':
    
    main()