import matplotlib.pyplot as plt

def plot_payoff(payoff, title = "Payoff matrix", folder = None, save = False):
    
    f = plt.figure(1)
    plt.title(title)
    plt.xlabel('Player 2')
    plt.ylabel('Player 1')
    plt.imshow(payoff)
    plt.colorbar()
    
    if save:
        plt.savefig('../WEEK 3/' + folder + '/' + title + '.png')
    else:
        f.show()

def plot_potential(mean_potential, title = None, folder = None, save = False):
    
    f = plt.figure(2)
    f.clf()
    plt.plot(mean_potential)
    plt.ylabel('Potential')
    plt.xlabel('Iteration [#]')
    
    if save:
        plt.savefig('../WEEK 3/' + folder + '/' + title + '.png')
    else:
        f.show()
        
    
def plot_potential_with_std(mean_potential, std, title = None, folder = None, save = False):
    
    f = plt.figure(3)
    f.clf()
    plt.plot(mean_potential)
    plt.fill_between(range(0, len(mean_potential)), mean_potential - std, mean_potential + std,
    alpha=0.2, edgecolor='#089FFF', facecolor='#089FFF', linewidth=4, antialiased=True)
    plt.ylabel('Potential')
    plt.xlabel('Iteration [#]')
    
    if save:
        plt.savefig('../WEEK 3/' + folder + '/' + title + '.png')
    else:
        f.show()

def plot_potential_history(potentials_history, title = None, folder = None, save = False):
    
    f = plt.figure(4)
    f.clf()
    for idx, potential in enumerate(potentials_history):
        plt.plot(potential, label = str(idx))

    plt.ylabel('Potential')
    plt.xlabel('Iteration [#]')
    
    if save:
        plt.savefig('../WEEK 3/' + folder + '/' + title + '.png')
    else:
        f.show()