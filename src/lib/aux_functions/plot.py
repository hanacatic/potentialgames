import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots
import networkx as nx 
import numpy as np

plt.rcParams.update({'figure.dpi': '100'})

# TODO have them all have step and add some comments?

def plot_payoff(payoff, title = "Payoff matrix", folder = None, save = False, file_name = "Payoff matrix"):
    
    with plt.style.context(["science"]):
        fig, ax = plt.subplots()
    
        ax.set_xlabel('Player 2')
        ax.set_ylabel('Player 1')
        plt.imshow(payoff)
        plt.colorbar()
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
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
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
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
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
        else:
            fig.show()

def plot_lines(lines_to_plot, list_labels, iter = None, plot_e_efficient = False, title = 'Average potential', file_name = None, folder = None, save = False):
      
    with plt.style.context(["science"]):
  
        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()
        
        t = np.arange(iter)
        for idx, element in enumerate(lines_to_plot):
            if idx < len(list_labels) - 1:
                ax.plot(t[::100], element[0:iter:100], label = list_labels[idx], markevery=10000)
            elif plot_e_efficient:
                ax.plot(element[0:iter], 'k--', label =  list_labels[idx])
            
        ax.set_xlabel('T')
        ax.set_ylabel('Expected potential value')
        ax.legend(fontsize = 'x-small', loc="lower right", frameon = True)

        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
        else:
            fig.show()
   
def plot_lines_with_std(lines_to_plot, std, list_labels, iter = None, step = 1, plot_e_efficient = False, conv_idx = None, title = 'Average potential', file_name = None, folder = None, save = False, legend = "lower right"):
    
    with plt.style.context(["ieee"]):

        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()
        
        colormap = cm.get_cmap("plasma") # The various colormaps can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

        t = np.arange(iter)
        N = len(lines_to_plot)
        for idx, element in enumerate(lines_to_plot):
            element = element[0:iter]
            if plot_e_efficient == False or (idx < len(lines_to_plot) - 1 and plot_e_efficient):
                one_std = std[idx][0:iter]
                line, = ax.plot(t[::step], element[::step], label = list_labels[idx], zorder = 2)

                line_colour = colormap(idx/N) # This function takes a number between 0 and 1 and returns a color.
                line.set_color(line_colour)
                ax.fill_between(t[::step], (element - one_std)[::step], (element + one_std)[::step],
        alpha=0.2, edgecolor=line_colour, facecolor=line_colour, linewidth=4, antialiased=True, zorder=2)
                if not conv_idx is None:
                    ax.scatter(conv_idx[idx+1], conv_idx[0], marker='*', s=40, color=line_colour, edgecolors=line_colour, zorder = 2)  # Star markers
            elif plot_e_efficient:
                print(idx)
                ax.plot(element, 'k--', label =  list_labels[idx], zorder = 1)
        
        ax.set_xlabel('T', fontsize = 6)
        ax.set_title('Expected potential value', fontsize = 8)
        ax.tick_params(axis='both', labelsize=6, length = 2)  
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.legend(fontsize='medium', loc="lower right", frameon = True)
        
        ax.set_xlim([0, iter])
        ax.set_ylim(0.0, 1.05)
        
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
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

        lines = ax.lines
        N = len(lines)
        for n in range(0, N, 2): # For each two-lines made via `plt.plot(...)`:
            random_color = colormap(n/N) # This function takes a number between 0 and 1 and returns a color.
            lines[n].set_color(random_color)
            lines[n+1].set_color(random_color)
            
        ax.set_xlabel('T')
        ax.legend(fontsize='x-small', loc="lower right")
        ax.ylim(0.6, 1.1)
        
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
        else:
            fig.show()

def plot_lines_eps_exp_with_std(lines_to_plot, std, list_labels, iter = None, step = 1, plot_e_efficient = False, conv_idx = None, title = 'Average potential', file_name = None, folder = None, save = False):
    
    with plt.style.context(["ieee"]):

        if iter is None:
            iter = len(lines_to_plot[0])
            
        fig, ax = plt.subplots()
        t = np.arange(iter)

        for idx in range(0, len(lines_to_plot)):
            ax.plot(t[::step], lines_to_plot[idx ][0:iter:step], label = list_labels[idx])
            
        colormap = cm.get_cmap("plasma") # The various colormaps can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        lines = ax.lines

        N = len(lines)
        for n in range(0, N): # For each two-lines made via `plt.plot(...)`:
            one_std = std[n][0:iter:step]
            element = lines_to_plot[n][0:iter:step]
            random_color = colormap(n/N) # This function takes a number between 0 and 1 and returns a color.
            lines[n].set_color(random_color)
            ax.fill_between(t[::step], element - one_std, element + one_std,
                            alpha=0.2, edgecolor=random_color, facecolor=random_color, linewidth=4, antialiased=True)

            if not conv_idx is None:
                ax.scatter(conv_idx[2*n], conv_idx[2*n+1], marker='*', s=40, color=random_color, zorder = 3)  # Star markers

        ax.set_xlabel('T', fontsize = 6)
        ax.set_title('Expected potential value', fontsize = 8)
        ax.tick_params(axis='both', labelsize=6, length = 2)  
        ax.xaxis.get_offset_text().set_fontsize(6)

        ax.legend(fontsize='medium', loc="lower right", frameon = True)
        
        ax.set_ylim(0.8, 1.05)
        ax.set_xlim(0, iter)
        if save:
            fig.savefig('../' + folder + '/' + file_name + '.png', dpi = 600)
            fig.savefig('../' + folder + '/' + file_name + '.svg', dpi = 600)
        else:
            fig.show()
          
def plot_network(network):
    
    nx.draw(network , nx.get_node_attributes(network, 'pos') , with_labels=True, node_size=500)
    nx.draw_networkx_edge_labels(network , nx.get_node_attributes(network, 'pos') , edge_labels = nx.get_edge_attributes(network,'weight'))

    plt.show()