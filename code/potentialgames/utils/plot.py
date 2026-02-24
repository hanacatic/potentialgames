import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scienceplots
import numpy as np
from pathlib import Path

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
    fig.savefig(f'{folder}/{file_name}.png', dpi=600)
    # fig.savefig(f'{folder}/{file_name}.svg', dpi=600)

def plot_matrix(matrix, xlabel='', ylabel='', title="Matrix", folder=None, file_name=None, save=False):
    """
    Plot a matrix as an image.

    Parameters:
        matrix (array-like): The matrix to plot.
        xlabel (str, optional): Label for the x-axis. Default is "" (empty).
        ylabel (str, optional): Label for the y-axis. Default is "" (empty).
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
            plt.show(block=False)
            plt.pause(10)


def plot_line(line, xlabel='', ylabel='', title='Line', lim=None, save=False, folder=None, file_name=None):
    """
    Plot a line over time.

    Parameters:
        line (array-like): The values to plot.
        xlabel (str): Label for the x-axis. Default is "" (empty).
        ylabel (str): Label for the y-axis. Default is "" (empty).
        title (str): Plot title. Default is "Line".
        lim (tuple, optional): y-axis limits as (min, max). Default is None.
        save (bool): Whether to save the plot or show it.
        folder (str): Folder to save the file in.
        file_name (str): Name for saving the file (without extension).
    """
    with plt.style.context(["science"]):
        
        fig, ax = plt.subplots()
        ax.plot(line)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        if lim is not None:
            ax.set_ylim(lim)
            
        if save:
            save_figure(fig, folder, file_name)
        else:
            plt.show(block=False)
            plt.pause(10)

def compare_lines(
    lines,
    lines_t=None,
    lines_legend=None,
    std=None,
    markers=None,
    special_lines=None,
    special_lines_legend=None,
    title='Lines',
    title_fontsize=7,
    xlabel='',
    ylabel='',
    label_fontsize=6,
    legend_fontsize='x-small',
    colormap="Dark2",
    xlim=None,
    ylim=None,
    save=False,
    folder=None,
    file_name=None
):
    """
    Plot multiple lines, optionally with special lines (e.g., efficient frontier).

    Parameters:
        lines (list of array-like): Data for each line.
        lines_legend (list of str): Labels for each line.
        std (list of array-like, optional): Standard deviation for each line. Default is None.
        special_lines (list of array-like, optional): Special lines to plot (e.g., efficient frontier). Default is None.
        special_lines_legend (list of str, optional): Labels for special lines. Default is None.
        title (str): Plot title. Default is "Lines".
        xlabel (str, optional): Label for the x-axis. Default is "" (empty).
        ylabel (str, optional): Label for the y-axis. Default is "" (empty).
        save (bool): Whether to save the plot or show it.
        folder (str): Folder to save the file in.
        file_name (str): Name for saving the file (without extension).
    """
    with plt.style.context(["ieee"]):
        
        fig, ax = plt.subplots()

        colormap = cm.get_cmap(colormap) 
        
        if lines_legend is None:
            lines_legend = [None] * len(lines)
        elif len(lines_legend) != len(lines):
            raise ValueError("Length of lines_legend must match length of lines.")

        for idx, element in enumerate(lines):
            if lines_t is not None:
                t = lines_t[idx]
            else:
                t = np.arange(len(element))
            line, = ax.plot(t, element, label=lines_legend[idx])
            line_colour = colormap(idx/len(lines))
            line.set_color(line_colour)
            
            if std is not None:
                one_std = std[idx]
                ax.fill_between(
                    t,
                    np.array(element) - np.array(one_std),
                    np.array(element) + np.array(one_std),
                    alpha=0.2,
                    edgecolor=line_colour, 
                    facecolor=line_colour, 
                    linewidth=4,
                    antialiased=True,
                    zorder=2
                )
            if markers is not None and markers[idx][0] is not None:
                ax.scatter(
                    markers[idx][0], 
                    markers[idx][1], 
                    marker='*', 
                    s=40, 
                    color=line_colour, 
                    edgecolors=line_colour, 
                    zorder=2
                )

        if special_lines is not None:
            
            if special_lines_legend is None:
                special_lines_legend = [None] * len(special_lines)
            elif len(special_lines_legend) != len(special_lines):
                raise ValueError("Length of special_lines_legend must match length of special_lines.")
                
            for idx, element in enumerate(special_lines):
                t = np.arange(len(element))
                ax.plot(t, element, 'k--', label=special_lines_legend[idx], zorder=1)

        ax.set_xlabel(xlabel, fontsize = label_fontsize)
        ax.set_ylabel(ylabel, fontsize = label_fontsize)
        ax.set_title(title, fontsize = title_fontsize)
        ax.tick_params(axis='both', length=2, labelsize=label_fontsize)
        ax.xaxis.get_offset_text().set_fontsize(label_fontsize)
        ax.legend(fontsize=legend_fontsize, loc="lower right", frameon=True)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if save:
            Path(folder).mkdir(parents=True, exist_ok=True) 
            save_figure(fig, folder, file_name)
        else:
            plt.show(block=False)
            plt.pause(20)
    
