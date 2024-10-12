import numpy as np
from plot import *
# def metropolis_hastings(prob_func, initial_profile, iterations = 100):
# in this implementation it is not working because the application is discrete
    
#     current_profile = initial_profile
    
#     for i in range(iterations):
#         proposed_profile = proposal_sampler(current_profile)
        
#         acceptance_ratio = prob_func(proposed_profile)/prob_func(current_profile)
        
#         if np.random.uniform(0, 1) < acceptance_ratio:
#             return proposed_profile
        
#     return -1

# def proposal_sampler(action_profile):
#     return action_profile + np.random.normal(0, 0.1, size = action_profile.shape)

def rejection_sampling(prob_func, initial_profile, action_space_size, M = 1.0, iterations = 1000):
    
    current_profile = initial_profile
    
    for _ in range(iterations):
        proposed_profile = proposal_sampler(current_profile, action_space_size)
        
        acceptance_ratio = prob_func(proposed_profile)/(M*prob_func(current_profile))
        
        if np.random.uniform(0, 1) < acceptance_ratio:
            return proposed_profile
        
    return -1

def proposal_sampler(action_profile, action_space_size):
    return np.random.randint(0, action_space_size, size = action_profile.shape)


def beta_experiments(game, n_exp = 10, eps = 0.1, save = False, folder = None, file_name = None, title = 'Average potential'): 
    
    beta_t = game.compute_beta(eps)
    # game.set_max_iter(EPS)
    print(beta_t)
    print(game.max_iter)
        
    plot_payoff(game.gameSetup.payoff_player_1, folder = folder, save = save)
    mean_potential_history = np.zeros((7, game.max_iter))
    betas = np.arange(beta_t/2, beta_t + beta_t/8.0, beta_t/8.0)
    
    for idx, beta in enumerate(betas):
        print(beta)
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history)
            game.play(beta=beta)
            
            game.gameSetup.reset_payoff_matrix()
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    game = Game(game.gameSetup, algorithm = "log_linear_t", mu=mu)
    game.set_initial_action_profile(game.initial_action_profile)
    
    potentials_history = np.zeros((n_exp, game.max_iter))

    for i in range(0, n_exp):
        game.play()
        potentials_history[i] = np.transpose(game.potentials_history)
   
    mean_potential_history[5] = np.mean(potentials_history, 0)

    mean_potential_history[6] = (1-EPS) * np.ones((1, game.max_iter))
    labels = [ r'$\frac{\beta_T}{2}$', r'$\frac{5\beta_T}{8}$', r'$\frac{6\beta_T}{8}$', r'$\frac{7\beta_T}{8}$', r'$\beta_T$', r'$\beta(t)$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title, file_name = file_name, save = save, folder = folder)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()
        
def delta_experiments(game, deltas = [0.9, 0.75, 0.5, 0.25, 0.1],  n_exp = 10, eps = 0.1, save = False, folder = None, file_name = None, title = 'Average potential'):
       
    beta_t = game.compute_beta(eps)
    plot_payoff(game.gameSetup.payoff_player_1)
    mean_potential_history = np.zeros((6, game.max_iter))
    
    for idx, delta in enumerate(deltas):
        
        print(delta)
        game.gameSetup.reset_payoff_matrix(delta)
        game.reset_game()
        plot_payoff(game.gameSetup.payoff_player_1)
        beta_t = game.compute_beta(eps)
        print(beta_t)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history).copy()
            game.play(beta=beta_t)
            
            game.gameSetup.reset_payoff_matrix(delta)
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
    mean_potential_history[5] = (1-eps) * np.ones((1, game.max_iter))
    labels = [ r'$\Delta = 0.9$', r'$\Delta = 0.75$', r'$\Delta = 0.5$', r'$\Delta = 0.25$', r'$\Delta = 0.1$', r'$\Phi(a^*) - \epsilon$']
   
    plot_lines(mean_potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        # plt.show()
        plt.show(block = False)
        plt.pause(20)
        plt.close()

def epsilon_experiments(game, epsilons = [0.2, 0.1, 0.05, 0.01, 0.001], n_exp = 10, save = False, folder = None, file_name = None, title = 'Average potential'):
    
    mean_potential_history = np.zeros((10, game.max_iter))
    
    for idx, eps in enumerate(epsilons):
        
        beta = game.compute_beta(eps)
        print(beta)
        
        potentials_history = np.zeros((n_exp, game.max_iter))
        for i in range(0, n_exp):
            potentials_history[i] = np.transpose(game.potentials_history)
            game.play(beta=beta)
            
            game.gameSetup.reset_payoff_matrix()
            game.reset_game()
            
        mean_potential_history[idx] = np.mean(potentials_history, 0)
    
        mean_potential_history[idx + 5] = (1 - eps) * np.ones((1, game.max_iter))
    labels = [ r'$\epsilon = 0.2$', r'$\epsilon = 0.1$', r'$\epsilon = 0.05$', r'$\epsilon = 0.01$', r'$\epsilon = 0.001$']
   
    plot_lines_eps_exp(mean_potential_history, labels, True, title = title, folder = folder, save = save, file_name = file_name)

    if not save:
        plt.show()
        # plt.show(block = False)
        # plt.pause(20)
        # plt.close()
