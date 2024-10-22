import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_payoff(payoff, title = "Payoff matrix", folder = None, save = False, file_name = "Payoff matrix"):
    
    f = plt.figure()
    plt.title(title)
    plt.xlabel('Player 2', fontsize=10)
    plt.ylabel('Player 1', fontsize=10)
    plt.title(title, fontsize=15)
    plt.imshow(payoff)
    plt.colorbar()
    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()

def plot_potential(mean_potential, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure()
    f.clf()
    plt.plot(mean_potential)
    plt.xlabel('time', fontsize=10)
    plt.title(title, fontsize=15)
    plt.ylim(0, 1.1)
    
    plt.grid()
    
    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()
        
    
def plot_potential_with_std(mean_potential, std, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure()
    f.clf()
    plt.plot(mean_potential)
    plt.fill_between(range(0, len(mean_potential)), mean_potential - std, mean_potential + std,
    alpha=0.2, edgecolor='#089FFF', facecolor='#089FFF', linewidth=4, antialiased=True)
    plt.xlabel('time', fontsize=10)
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.grid()

    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()
        
def plot_lines_with_std(lines_to_plot, std, list_labels, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure()
    f.clf()
    
    for idx, element in enumerate(lines_to_plot):
        if plot_e_efficient == False or (idx < len(lines_to_plot) - 1 and plot_e_efficient):
            line, = plt.plot(element, label = list_labels[idx])
            line_colour = line.get_color()
            plt.fill_between(range(0, len(element)), element - std[idx], element + std[idx],
    alpha=0.2, edgecolor=line_colour, facecolor=line_colour, linewidth=4, antialiased=True)
        elif plot_e_efficient:
            plt.plot(element, 'k--', label =  list_labels[idx])
        
        
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
        
def plot_lines(lines_to_plot, list_labels, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure()
    f.clf()
    
    for idx, element in enumerate(lines_to_plot):
        if idx < len(lines_to_plot) - 1:
            plt.plot(element, label = list_labels[idx])
        elif plot_e_efficient:
            plt.plot(element, 'k--', label =  list_labels[idx])
        
        
    # plt.xlabel('Potential', fontsize=15)
    plt.xlabel('time', fontsize=10)
    plt.legend(fontsize=15)
    plt.title(title, fontsize=15)
    plt.ylim(0, 1.1)
    plt.legend(loc="lower right")
    plt.grid()
    if save:
        plt.savefig('../' + folder + '/' + file_name + '.png')
    else:
        f.show()

def plot_lines_eps_exp(lines_to_plot, list_labels, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
    
    f = plt.figure()
    f.clf()
    
    for idx in range(0, int(len(lines_to_plot)/2)):
        plt.plot(lines_to_plot[idx ], label = list_labels[idx])
        plt.plot(lines_to_plot[idx + int(len(lines_to_plot)/2)], 'k--')

    colormap = cm.get_cmap("plasma") # The various colormaps can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    ax = plt.gca()
    lines = ax.lines
    N = len(lines)
    for n in range(0, N, 2): # For each two-lines made via `plt.plot(...)`:
        random_color = colormap(n/N) # This function takes a number between 0 and 1 and returns a color.
        lines[n].set_color(random_color)
        lines[n+1].set_color(random_color)
        
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