import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots
import networkx as nx 

def plot_payoff(payoff, title = "Payoff matrix", folder = None, save = False, file_name = "Payoff matrix"):
    
    with plt.style.context(["science"]):
        fig, ax = plt.subplots()
    
        ax.set_xlabel('Player 2')
        ax.set_ylabel('Player 1')
        plt.imshow(payoff)
        plt.colorbar()
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()

def plot_potential(mean_potential, title = 'Average potential', file_name = None, folder = None, save = False):
    
    with plt.style.context(["science"]):
        fig, ax = plt.subplots()

        ax.plot(mean_potential)
        ax.set_ylabel('Expected potential value')
        ax.set_xlabel('T')
        ax.set_ylim(0.8, 1.1)
                
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()
        
def plot_potential_with_std(mean_potential, std, title = 'Average potential', file_name = None, folder = None, save = False):
    
    with plt.style.context(["science"]):
        
        fig, ax = plt.subplots()
        ax.plot(mean_potential)
        ax.fill_between(range(0, len(mean_potential)), mean_potential - std, mean_potential + std,
        alpha=0.2, edgecolor='#089FFF', facecolor='#089FFF', linewidth=4, antialiased=True)
        ax.set_xlabel('T')
        ax.set_ylabel('Expected potential value')
        ax.set_ylim(0.8, 1.1)

        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()

def plot_lines(lines_to_plot, list_labels, iter = None, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
      
    with plt.style.context(["science"]):
  
        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()
        
        for idx, element in enumerate(lines_to_plot):
            if idx < len(list_labels) - 1:
                ax.plot(element[0:iter], label = list_labels[idx])
            elif plot_e_efficient:
                ax.plot(element[0:iter], 'k--', label =  list_labels[idx])
            
        ax.set_xlabel('T')
        ax.set_ylabel('Expected potential value')
        ax.legend(fontsize = 'x-small', loc="lower right", frameon = True)

        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()
   
def plot_lines_with_std(lines_to_plot, std, list_labels, iter = None, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False, legend = "lower right"):
    
    with plt.style.context(["science"]):

        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()
        
        for idx, element in enumerate(lines_to_plot):
            element = element[0:iter]
            if plot_e_efficient == False or (idx < len(lines_to_plot) - 1 and plot_e_efficient):
                one_std = std[idx][0:iter]
                line, = ax.plot(element, label = list_labels[idx])
                line_colour = line.get_color()
                ax.fill_between(range(0, len(element)), element - one_std, element + one_std,
        alpha=0.2, edgecolor=line_colour, facecolor=line_colour, linewidth=4, antialiased=True)
            elif plot_e_efficient:
                print(idx)
                ax.plot(element, 'k--', label =  list_labels[idx])
            
        ax.set_xlabel('T')
        ax.set_ylabel('Expected potential value')
        ax.legend(fontsize = 'x-small', loc = legend, frameon = True)
        
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()
        
def plot_lines_eps_exp(lines_to_plot, list_labels, iter = None, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
    
    with plt.style.context(["science"]):

        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()
        
        for idx in range(0, int(len(lines_to_plot)/2)):
            ax.plot(lines_to_plot[idx, 0:iter ], label = list_labels[idx])
            ax.plot(lines_to_plot[idx + int(len(lines_to_plot)/2), 0:iter], 'k--')

        colormap = cm.get_cmap("plasma") # The various colormaps can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        # ax = plt.gca()
        lines = ax.lines
        N = len(lines)
        for n in range(0, N, 2): # For each two-lines made via `plt.plot(...)`:
            random_color = colormap(n/N) # This function takes a number between 0 and 1 and returns a color.
            lines[n].set_color(random_color)
            lines[n+1].set_color(random_color)
            
        # plt.xlabel('Potential', fontsize=15)
        ax.set_xlabel('T')
        ax.legend(fontsize='x-small', loc="lower right")
        ax.ylim(0.6, 1.1)
        
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()

def plot_lines_eps_exp_with_std(lines_to_plot, std, list_labels, iter = None, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
    
    with plt.style.context(["science"]):

        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()

    
        for idx in range(0, int(len(lines_to_plot)/2)):
            ax.plot(lines_to_plot[idx ][0:iter], label = list_labels[idx])
            ax.plot(lines_to_plot[idx + int(len(lines_to_plot)/2)][0:iter], 'k--')
            
        colormap = cm.get_cmap("plasma") # The various colormaps can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        # ax = plt.gca()
        lines = ax.lines
        N = len(lines)
        for n in range(0, N, 2): # For each two-lines made via `plt.plot(...)`:
            one_std = std[int(n/2)][0:iter]
            element = lines_to_plot[int(n/2)][0:iter]
            random_color = colormap(n/N) # This function takes a number between 0 and 1 and returns a color.
            lines[n].set_color(random_color)
            lines[n+1].set_color(random_color)
            ax.fill_between(range(0, len(element)), element - one_std, element + one_std,
                            alpha=0.2, edgecolor=random_color, facecolor=random_color, linewidth=4, antialiased=True)

            
            
        # plt.xlabel('Potential', fontsize=15)
        ax.set_xlabel('T')
        ax.legend(fontsize='x-small', loc="lower right")
        ax.ylim(0.6, 1.1)
        
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 300)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 300)
        else:
            fig.show()
          
def plot_network(network):
    
    nx.draw(network , nx.get_node_attributes(network, 'pos') , with_labels=True, node_size=500)
    nx.draw_networkx_edge_labels(network , nx.get_node_attributes(network, 'pos') , edge_labels = nx.get_edge_attributes(network,'weight'))

    plt.show()