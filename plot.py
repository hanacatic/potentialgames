import matplotlib.pyplot as plt

def plot_payoff(payoff, title = "Payoff matrix", folder = None, save = False):
    
    f = plt.figure(1)
    plt.title(title)
    plt.xlabel('Player 2', fontsize=10)
    plt.ylabel('Player 1', fontsize=10)
    plt.title(title, fontsize=15)
    plt.imshow(payoff)
    plt.colorbar()
    
    if save:
        plt.savefig('../' + folder + '/' + title + '.png')
    else:
        f.show()

def plot_potential(mean_potential, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure(2)
    f.clf()
    plt.plot(mean_potential)
    plt.xlabel('time', fontsize=10)
    plt.title(title, fontsize=15)
    plt.ylim(0, 1.1)
    
    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()
        
    
def plot_potential_with_std(mean_potential, std, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure(3)
    f.clf()
    plt.plot(mean_potential)
    plt.fill_between(range(0, len(mean_potential)), mean_potential - std, mean_potential + std,
    alpha=0.2, edgecolor='#089FFF', facecolor='#089FFF', linewidth=4, antialiased=True)
    plt.xlabel('time', fontsize=10)
    plt.title(title)
    plt.ylim(0, 1.1)

    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()

def plot_lines(lines_to_plot, list_labels, title, file_name = None, folder = None, save = False):
    
    f = plt.figure()
    f.clf()
    for idx, element in enumerate(lines_to_plot):
        plt.plot(element, label = list_labels[idx])

    # plt.xlabel('Potential', fontsize=15)
    plt.xlabel('time', fontsize=10)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")
    
    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()