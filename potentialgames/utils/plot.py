import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots
import networkx as nx 
import numpy as np

plt.rcParams.update({'figure.dpi': '100'})

def save_figure(fig, folder, file_name):
    """
    Save a matplotlib figure in both PNG and SVG formats.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to save.
        folder (str): Folder to save the file in.
        file_name (str): Name for saving the file (without extension).
    """
    if folder is None or file_name is None:
        raise ValueError("Both folder and file_name must be specified to save the figure.")
    fig.savefig(f'../{folder}/{file_name}.png', dpi=600)
    fig.savefig(f'../{folder}/{file_name}.svg', dpi=600)

def plot_matrix(matrix, xlabel, ylabel, title="Matrix", folder=None, file_name=None, save=False):
    """
    Plot a matrix as an image.

    Parameters:
        matrix (array-like): The matrix to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str, optional): Title for the plot. Default is "Matrix".
        folder (str, optional): Folder to save the file in. Default is None.
        file_name (str, optional): Name for saving the file (without extension). Default is None.
        save (bool, optional): Whether to save the plot (True) or show it (False). Default is False.
    """
    with plt.style.context(["science"]):
        fig, ax = plt.subplots()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.imshow(matrix)
        plt.colorbar()
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()

def plot_potential(mean_potential, title='Average potential', file_name=None, folder=None, save=False):
    """
    Plot the mean potential over time.

    Parameters:
        mean_potential (array-like): The mean potential values.
        title (str): Plot title (currently unused).
        file_name (str): Name for saving the file (without extension).
        folder (str): Folder to save the file in.
        save (bool): Whether to save the plot or show it.
    """
    with plt.style.context(["science"]):
        fig, ax = plt.subplots()
        ax.plot(mean_potential)
        ax.set_ylabel('Expected potential value')
        ax.set_xlabel('T')
        ax.set_ylim(0.8, 1.1)
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()

def plot_potential_with_std(mean_potential, std, title='Average potential', file_name=None, folder=None, save=False):
    """
    Plot the mean potential with standard deviation shading.

    Parameters:
        mean_potential (array-like): The mean potential values.
        std (array-like): Standard deviation values.
        title (str): Plot title (currently unused).
        file_name (str): Name for saving the file (without extension).
        folder (str): Folder to save the file in.
        save (bool): Whether to save the plot or show it.
    """
    with plt.style.context(["science"]):
        fig, ax = plt.subplots()
        ax.plot(mean_potential)
        ax.fill_between(range(0, len(mean_potential)), mean_potential - std, mean_potential + std,
                        alpha=0.2, edgecolor='#089FFF', facecolor='#089FFF', linewidth=4, antialiased=True)
        ax.set_xlabel('T')
        ax.set_ylabel('Expected potential value')
        ax.set_ylim(0.8, 1.1)
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()

def plot_lines(lines_to_plot, list_labels, iter=None, plot_e_efficient=False, title='Average potential', file_name=None, folder=None, save=False):
    """
    Plot multiple lines, optionally with an efficient frontier.

    Parameters:
        lines_to_plot (list of array-like): Data for each line.
        list_labels (list of str): Labels for each line.
        iter (int): Number of iterations to plot.
        plot_e_efficient (bool): Whether to plot the efficient frontier.
        title (str): Plot title (currently unused).
        file_name (str): Name for saving the file (without extension).
        folder (str): Folder to save the file in.
        save (bool): Whether to save the plot or show it.
    """
    with plt.style.context(["science"]):
        if iter is None:
            iter = len(lines_to_plot[0])
        fig, ax = plt.subplots()
        t = np.arange(iter)
        for idx, element in enumerate(lines_to_plot):
            if idx < len(list_labels) - 1:
                ax.plot(t[::100], element[0:iter:100], label=list_labels[idx], markevery=10000)
            elif plot_e_efficient:
                ax.plot(element[0:iter], 'k--', label=list_labels[idx])
        ax.set_xlabel('T')
        ax.set_ylabel('Expected potential value')
        ax.legend(fontsize='x-small', loc="lower right", frameon=True)
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()
   
def plot_lines_with_std(lines_to_plot, std, list_labels, iter=None, step=1, plot_e_efficient=False, conv_idx=None, title='Average potential', file_name=None, folder=None, save=False, legend="lower right"):
    """
    Plot multiple lines with standard deviation shading, optionally with an efficient frontier.

    Parameters:
        lines_to_plot (list of array-like): Data for each line.
        std (list of array-like): Standard deviation for each line.
        list_labels (list of str): Labels for each line.
        iter (int): Number of iterations to plot.
        step (int): Step size for plotting.
        plot_e_efficient (bool): Whether to plot the efficient frontier.
        conv_idx (list or None): Indices for convergence markers.
        title (str): Plot title (currently unused).
        file_name (str): Name for saving the file (without extension).
        folder (str): Folder to save the file in.
        save (bool): Whether to save the plot or show it.
        legend (str): Legend location.
    """
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
                line, = ax.plot(t[::step], element[::step], label=list_labels[idx], zorder=2)
                line_colour = colormap(idx/N)
                line.set_color(line_colour)
                ax.fill_between(t[::step], (element - one_std)[::step], (element + one_std)[::step],
                               alpha=0.2, edgecolor=line_colour, facecolor=line_colour, linewidth=4, antialiased=True, zorder=2)
                if not conv_idx is None:
                    ax.scatter(conv_idx[idx+1], conv_idx[0], marker='*', s=40, color=line_colour, edgecolors=line_colour, zorder=2)
            elif plot_e_efficient:
                print(idx)
                ax.plot(element, 'k--', label=list_labels[idx], zorder=1)
        ax.set_xlabel('T', fontsize=6)
        ax.set_title('Expected potential value', fontsize=8)
        ax.tick_params(axis='both', labelsize=6, length=2)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.legend(fontsize='medium', loc=legend, frameon=True)
        ax.set_xlim([0, iter])
        ax.set_ylim(0.0, 1.05)
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()
        
def plot_lines_eps_exp(lines_to_plot, list_labels, iter=None, plot_e_efficient=False, title='Average potential', file_name=None, folder=None, save=False):
    """
    Plot pairs of lines (e.g., expected value and its epsilon-approximation) for each label.

    Parameters:
        lines_to_plot (array-like): 2D array, each pair of rows is a line and its epsilon-approximation.
        list_labels (list of str): Labels for each line.
        iter (int): Number of iterations to plot.
        plot_e_efficient (bool): Whether to plot the efficient frontier (currently unused).
        title (str): Plot title (currently unused).
        file_name (str): Name for saving the file (without extension).
        folder (str): Folder to save the file in.
        save (bool): Whether to save the plot or show it.
    """
    with plt.style.context(["science"]):
        if iter is None:
            iter = len(lines_to_plot[0])
        fig, ax = plt.subplots()
        for idx in range(0, int(len(lines_to_plot)/2)):
            ax.plot(lines_to_plot[idx, 0:iter], label=list_labels[idx])
            ax.plot(lines_to_plot[idx + int(len(lines_to_plot)/2), 0:iter], 'k--')

        colormap = cm.get_cmap("plasma") # The various colormaps can be found here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

        lines = ax.lines
        N = len(lines)
        for n in range(0, N, 2):
            random_color = colormap(n/N)
            lines[n].set_color(random_color)
            lines[n+1].set_color(random_color)
        ax.set_xlabel('T')
        ax.legend(fontsize='x-small', loc="lower right")
        ax.ylim(0.6, 1.1)
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()

def plot_lines_eps_exp_with_std(lines_to_plot, std, list_labels, iter=None, step=1, plot_e_efficient=False, conv_idx=None, title='Average potential', file_name=None, folder=None, save=False):
    """
    Plot pairs of lines with standard deviation shading.

    Parameters:
        lines_to_plot (array-like): 2D array, each row is a line.
        std (array-like): Standard deviation for each line.
        list_labels (list of str): Labels for each line.
        iter (int): Number of iterations to plot.
        step (int): Step size for plotting.
        plot_e_efficient (bool): Whether to plot the efficient frontier (currently unused).
        conv_idx (list or None): Indices for convergence markers.
        title (str): Plot title (currently unused).
        file_name (str): Name for saving the file (without extension).
        folder (str): Folder to save the file in.
        save (bool): Whether to save the plot or show it.
    """
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
        for n in range(0, N):
            one_std = std[n][0:iter:step]
            element = lines_to_plot[n][0:iter:step]
            random_color = colormap(n/N)
            lines[n].set_color(random_color)
            ax.fill_between(t[::step], element - one_std, element + one_std,
                            alpha=0.2, edgecolor=random_color, facecolor=random_color, linewidth=4, antialiased=True)
            if not conv_idx is None:
                ax.scatter(conv_idx[2*n], conv_idx[2*n+1], marker='*', s=40, color=random_color, zorder=3)
        ax.set_xlabel('T', fontsize=6)
        ax.set_title('Expected potential value', fontsize=8)
        ax.tick_params(axis='both', labelsize=6, length=2)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.legend(fontsize='medium', loc="lower right", frameon=True)
        ax.set_ylim(0.8, 1.05)
        ax.set_xlim(0, iter)
        if save:
            save_figure(fig, folder, file_name)
        else:
            fig.show()
          
def plot_network(network):
    """
    Plot a networkx graph with node positions and edge weights.

    Parameters:
        network (networkx.Graph): The networkx graph to plot.
    """
    nx.draw(network, nx.get_node_attributes(network, 'pos'), with_labels=True, node_size=500)
    nx.draw_networkx_edge_labels(network, nx.get_node_attributes(network, 'pos'), edge_labels=nx.get_edge_attributes(network, 'weight'))
    plt.show()