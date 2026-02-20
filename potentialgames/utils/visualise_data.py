import numpy as np

import pickle
from pathlib import Path

from potentialgames.utils import compare_lines


def visualise_delta_experiment(algorithm, deltas = [0.15, 0.1, 0.075], eps = 0.05, iter = 1500000, title = "Expected potential value"):
    """
    Visualise the results of delta experiment obtained with function delta_experiment in potentialgames/scripts/delta_experiments.py.
    
    The function loads the potential history of the experiments from a pickle file, computes the mean and standard deviation across experiments, 
    and plots the expected potential value over time for different delta values. It also marks the convergence point where the expected potential 
    value exceeds 1-epsilon. The resulting plot can be saved to a specified directory.

    Args:
        algorithm (_type_): The learning algorithm used in the experiments (e.g., "fast_log_linear").
        deltas (list, optional): The suboptimality gap (delta) values for which to visualise the experiments. Defaults to [0.15, 0.1, 0.075].
        eps (float, optional): The tolerance value (epsilon) for convergence. Defaults to 0.05.
        iter (int, optional): The number of iterations to visualise. Defaults to 1500000.
        title (str, optional): The title of the plot. Defaults to "Expected potential value".
    """
    repo_root  = Path(__file__).resolve().parents[2]  
    data_path = repo_root / "data" / "experiments" / "delta"
    
    file_name = f"{algorithm}_potentials.pckl"
    
    legend = [fr'$\Delta = {delta}$' for delta in deltas] 

    save_root = repo_root / "data" / "visualisations" / "delta"
    save_file_name = f"{algorithm}_delta_experiment"
    
    visualise_data(data_path, [file_name], {
        "title": title, 
        "title_fontsize": 8,
        "xlabel": "T", 
        "ylabel": " ", 
        "xlim": [0, iter],
        "ylim": [0.8, 1.05],
        "legend": legend, 
        "legend_fontsize": 'medium',
        "colormap": "Dark2",
        "conv_val": [1-eps]*3, 
        "special_lines_legend": [fr'$\Phi(a^*) - \epsilon$']*3},
        save_root=save_root,
        save_file_name=save_file_name
    )
    
def visualise_epsilon_experiment(algorithm, epsilons = [0.1, 0.05, 0.025, 0.01], delta = 0.1, iter = 4500000, title = "Expected potential value"):
    """
    Visualise the results of epsilon experiment obtained with function epsilon_experiment in potentialgames/scripts/epsilon_experiments.py.
    
    The function loads the potential history of the experiments from a pickle file, computes the mean and standard deviation across experiments, 
    and plots the expected potential value over time for different epsilon values. It also marks the convergence point where the expected potential
    value exceeds 1-epsilon. The resulting plot can be saved to a specified directory.

    Args:
        algorithm (_type_): The learning algorithm used in the experiments (e.g., "fast_log_linear").
        epsilons (list, optional): The tolerance values (epsilon) for which to visualise the experiments. Defaults to [0.1, 0.05, 0.025, 0.01].
        delta (float, optional): The suboptimality gap (delta) value for which to visualise the experiments. Defaults to 0.1.
        iter (int, optional): The number of iterations to visualise. Defaults to 4500000.
        title (str, optional): The title of the plot. Defaults to "Expected potential value".
    """
    repo_root  = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "experiments" / "epsilon"
    
    file_name = f"{algorithm}_potentials.pckl"

    legend = [fr'$\epsilon = {epsilon}$' for epsilon in epsilons]
    conv_val = [1-epsilon for epsilon in epsilons]
    save_root = repo_root / "data" / "visualisations" / "epsilon"
    save_file_name = f"{algorithm}_epsilon_experiment"
    
    visualise_data(data_path, [file_name], {
        "title": title, 
        "title_fontsize": 9,
        "xlabel": "T", 
        "ylabel": " ", 
        "xlim": [0, iter],
        "ylim": [0.8, 1.05],
        "legend": legend, 
        "legend_fontsize": 'medium',
        "colormap": "plasma", 
        "conv_val": conv_val, 
        "special_lines_legend": None
        },
        save_root=save_root,
        save_file_name=save_file_name
    )

def visualise_full_feedback_comparison(iter = 200000, epsilon = 0.05, title="Expected potential value"):
    """
    Visualise the results of full feedback comparison obtained with function full_feedback_comparison in potentialgames/scripts/full_feedback_comparison_experiments.py.
    
    The function loads the potential history of the experiments for different algorithms from pickle files, computes the mean and standard deviation across experiments,
    and plots the expected potential value over time for each algorithm. It also marks the convergence point where the expected potential value exceeds 1-epsilon.

    Args:
        iter (int, optional): The number of iterations to visualise. Defaults to 200000.
        epsilon (float, optional): The tolerance value (epsilon) for which to visualise the experiments. Defaults to 0.05.
        title (str, optional): The title of the plot. Defaults to "Expected potential value".
    """
    algorithms = ["fast_log_linear", "hedge"]
    
    repo_root  = Path(__file__).resolve().parents[2] 
    data_path = repo_root / "data" / "experiments" / "full_feedback_comparison"

    file_names = [f"{algorithm}_potentials.pckl" for algorithm in algorithms]

    legend = [r"Log-linear learning ($\beta = \tilde{\Omega}(\frac{1}{\Delta}\log{\frac{A^N}{\epsilon}}) $)", "Hedge"]

    conv_val = [1-epsilon]*len(algorithms)
    save_root = repo_root / "data" / "visualisations" / "full_feedback_comparison"
    save_file_name = f"full_feedback_comparison_experiment"

    visualise_data(data_path, file_names, {
        "title": title,
        "title_fontsize": 9,
        "xlabel": "T",
        "ylabel": " ",
        "xlim": [0, iter],
        "ylim": [0.6, 1.05],
        "legend": legend,
        "legend_fontsize": 'small',
        "colormap": "Dark2",
        "conv_val": conv_val,
        "special_lines_legend": [fr'$\Phi(a^*) - \epsilon$']*len(algorithms)
        },
        save_root=save_root,
        save_file_name=save_file_name
    )

def visualise_delta_estimation_experiment(algorithm, est = [1, 1.5, 2., 2.5, 3., 3.5], delta = 0.1, epsilon = 0.05, iter = 10000, title = "Expected potential value"):
    """
    Visualise the results of experiments with estimated deltas obtained with function delta_estimation_experiment in potentialgames/scripts/delta_estimation_experiments.py.
    
    The function loads the potential history of the experiments for different estimated delta values from a pickle file, computes the mean and standard deviation across experiments,
    and plots the expected potential value over time for each estimated delta value. It also marks the convergence point where the expected potential value exceeds 1-epsilon.

    Args:
        algorithm (_type_): The learning algorithm used in the experiments (e.g., "fast_log_linear").
        est (list, optional): The list of estimated suboptimality gap (delta) values to visualise. Defaults to [1, 1.5, 2., 2.5, 3., 3.5].
        delta (float, optional): The true suboptimality gap (delta) value used in the experiments. Defaults to 0.1.
        epsilon (float, optional): The tolerance value (epsilon) for which to visualise the experiments. Defaults to 0.05.
        iter (int, optional): The number of iterations to visualise. Defaults to 10000.
        title (str, optional): The title of the plot. Defaults to "Expected potential value".
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "experiments" / "delta_est"

    file_name = f"{algorithm}_potentials.pckl"

    legend = [fr'$\hat{{\Delta}} = {e}\cdot\Delta$' for e in est]
    
    conv_val = [1-epsilon]*len(est)
    save_root = repo_root / "data" / "visualisations" / "delta_est"
    save_file_name = f"{algorithm}_delta_estimation_experiment"
    
    visualise_data(data_path, [file_name],{
        "title": title,
        "title_fontsize": 8,
        "xlabel": "T",
        "ylabel": " ",
        "xlim": [0, iter],
        "ylim": [0.8, 1.05],
        "legend": legend,
        "legend_fontsize": 'x-small',
        "colormap": "plasma",
        "conv_val": conv_val,
        "special_lines_legend": [fr'$\Phi(a^*) - \epsilon$']*len(est)
        },
        save_root=save_root,
        save_file_name=save_file_name
    )

def visualise_reduced_feedback_comparison(iter=20000, epsilon = 0.05, title="Expected potential value"):
    """
    Visualise the results of reduced feeback comparison obtained with function reduced_feedback_comparison in potentialgames/scripts/reduced_feedback_comparison.py.
    
    The function loads the potential history of the experiments for different algorithms from pickle files, computes the mean and standard deviation across experiments,
    and plots the expected potential value over time for each algorithm. It also marks the convergence point where the expected potential value exceeds 1-epsilon.

    Args:
        iter (int, optional): The number of iterations to visualise. Defaults to 20000.
        epsilon (float, optional): The tolerance value (epsilon) for which to visualise the experiments. Defaults to 0.05.
        title (str, optional): The title of the plot. Defaults to "Expected potential value".
    """
    algorithms = ["fast_binary_log_linear", "exp3p", "exponential_weight_with_annealing"]

    repo_root  = Path(__file__).resolve().parents[2]  # Adjust .parent levels if needed
    data_path = repo_root / "data" / "experiments" / "reduced_feedback_comparison"

    file_names = [f"{algorithm}_potentials.pckl" for algorithm in algorithms]

    legend = [r"Binary log-linear ($\beta = \tilde{\Omega}(\frac{1}{\Delta}\log{\frac{A^N}{\epsilon}}) $)", "EXP3", "Exponential weight with annealing"]

    conv_val = [1-epsilon]*len(algorithms)
    save_root = repo_root / "data" / "visualisations" / "reduced_feedback_comparison"
    save_file_name = f"reduced_feedback_comparison_experiment"

    visualise_data(data_path, file_names, {
        "title": title,
        "title_fontsize": 9,
        "xlabel": "T",
        "ylabel": " ",
        "xlim": [0, iter],
        "ylim": [0.6, 1.05],
        "legend": legend,
        "legend_fontsize": 'small',
        "colormap": "Dark2",
        "conv_val": conv_val,
        "special_lines_legend": [fr'$\Phi(a^*) - \epsilon$']*len(algorithms)
        },
        save_root=save_root,
        save_file_name=save_file_name
    )

def visualise_data(folder_root, file_names, meta_data, save_root=None, save_file_name=None):
    """
    Helper function to visualise the results of experiments. It loads the potential history of the experiments from pickle files, computes the mean and standard deviation across experiments,
    and plots the expected potential value over time for each algorithm. It also marks the convergence point where the expected potential value exceeds 1-epsilon. The resulting plot can be saved to a specified directory.
    
     Args:
        folder_root (Path): The root directory where the pickle files containing the potential history of the experiments are stored.
        file_names (list): A list of file names (pickle files) to load the potential history from.
        meta_data (dict): A dictionary containing metadata for the plot, including:         
            - "title": The title of the plot.
            - "title_fontsize": The font size of the title.
            - "xlabel": The label for the x-axis.
            - "ylabel": The label for the y-axis.
            - "xlim": The limits for the x-axis.
            - "ylim": The limits for the y-axis.
            - "legend": A list of legend labels for each algorithm.
            - "legend_fontsize": The font size of the legend.
            - "colormap": The colormap to use for the lines in the plot.
            - "conv_val": A list of convergence values (1-epsilon) for each algorithm to mark the convergence point on the plot.
            - "special_lines_legend": A list of legend labels for the special lines (e.g., convergence thresholds) to be plotted.
        save_root (Path, optional): The directory where the resulting plot should be saved. If None, the plot will not be saved. Defaults to None.
        save_file_name (str, optional): The name of the file to save the plot as. Defaults to None.
    """

    mean_data = np.empty((len(file_names), len(meta_data["legend"])))
    t_data = np.empty((len(file_names), len(meta_data["legend"])))
    std_data = np.empty((len(file_names), len(meta_data["legend"])))
    conv_markers = np.empty((len(file_names), len(meta_data["legend"]), 2))

    for idx, file_name in enumerate(file_names):

        file_path = folder_root / file_name
        
        with open(file_path, 'rb') as f:
            
            data = pickle.load(f)
                        
        mean = np.mean(data, axis=1)
        std = np.zeros(mean.shape)
        
        for i in range(data.shape[0]):
            
            std[i] = np.std(data[i], axis=0)
            
        # Reshape mean_data to match the shape of mean
        if mean_data.ndim == 2:
            
            c = 1
    
            if "fast" not in file_name:
                c = 2000
        
            mean_data = np.empty((len(file_names), mean.shape[0], int(mean.shape[1]/c)))
            std_data = np.empty((len(file_names), std.shape[0], int(std.shape[1]/c)))
            t_data = np.empty((len(file_names), mean.shape[0], int(mean.shape[1]/c)))
            conv_markers = np.empty((len(file_names), mean.shape[0], 2))
                    
        mean_data[idx] = mean[:, ::c]
        std_data[idx] = std[:, ::c]
        t_data[idx] = np.vstack([np.arange(mean.shape[1])[::c]] * mean.shape[0])
        
        for r in range(mean_data.shape[1]):
            conv_indices = np.argwhere(mean_data[idx, r] > meta_data['conv_val'][r])
            conv_markers[idx, r, 0] = t_data[idx, r, conv_indices[0][0]] if len(conv_indices) > 0 else None
            conv_markers[idx, r, 1] = mean_data[idx, r, conv_indices[0][0]] if len(conv_indices) > 0 else None
    
    mean_data = np.squeeze(mean_data)
    std_data = np.squeeze(std_data)
    conv_markers = np.squeeze(conv_markers)
    t_data = np.squeeze(t_data)
    
    print(conv_markers[:,0])

    special_lines = np.ones((mean_data.shape[0], mean_data.shape[1]*c))

    for i in range(len(meta_data['conv_val'])):
        special_lines[i] = meta_data['conv_val'][i] * special_lines[i]

    special_lines, unique_indices = np.unique(special_lines, axis=0, return_index=True)

    if meta_data["special_lines_legend"] is not None:
        special_lines_legend = np.array(meta_data["special_lines_legend"])[unique_indices]
    else:
        special_lines_legend = None
        
    save = False
    
    if len(file_names) == 1:
        prefix = file_names[0].split(".")[0]
    else:
        prefix = "comparison"
        
    mean_root = folder_root / f"{prefix}_mean_data.pckl"
    std_root = folder_root / f"{prefix}_std_data.pckl"
    conv_markers_root = folder_root / f"{prefix}_conv_markers.pckl"
    t_root = folder_root / f"{prefix}_t_data.pckl"
    meta_root = folder_root / f"{prefix}_meta_data.pckl"
    
    with open(mean_root, 'wb') as f:
        pickle.dump(mean_data, f, pickle.HIGHEST_PROTOCOL)
    with open(std_root, 'wb') as f:
        pickle.dump(std_data, f, pickle.HIGHEST_PROTOCOL)
    with open(conv_markers_root, 'wb') as f:
        pickle.dump(conv_markers, f, pickle.HIGHEST_PROTOCOL)
    with open(t_root, 'wb') as f:
        pickle.dump(t_data, f, pickle.HIGHEST_PROTOCOL)
    with open(meta_root, 'wb') as f:
        pickle.dump(meta_data, f, pickle.HIGHEST_PROTOCOL)
    
    if save_root is not None:
        save = True
                
    compare_lines(mean_data, t_data, meta_data["legend"], std_data, 
                  special_lines=special_lines, special_lines_legend=special_lines_legend, 
                  xlabel=meta_data["xlabel"], ylabel=meta_data["ylabel"], 
                  legend_fontsize=meta_data["legend_fontsize"], 
                  colormap=meta_data["colormap"],
                  xlim=meta_data["xlim"], ylim=meta_data["ylim"],
                  title=meta_data["title"], title_fontsize=meta_data["title_fontsize"],
                  markers=conv_markers, save=save, folder=save_root, file_name=save_file_name)
    
def visualise_experiment(experiment, algorithm, save_root=None, save_file_name=None):
    """Visualise preprocessed experiment data."""
    
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data" / "experiments" / experiment
    
    if algorithm == "comparison":
        suffix = f"comparison"
    else:
        suffix = f"{algorithm}_potentials"
    
    mean_path = data_path / f"{suffix}_mean_data.pckl"
    std_path = data_path / f"{suffix}_std_data.pckl"
    conv_markers_path = data_path / f"{suffix}_conv_markers.pckl"
    t_path = data_path / f"{suffix}_t_data.pckl"
    meta_path = data_path / f"{suffix}_meta_data.pckl"
    
    with open(mean_path, 'rb') as f:
        mean_data = pickle.load(f)
    with open(std_path, 'rb') as f:
        std_data = pickle.load(f)
    with open(conv_markers_path, 'rb') as f:
        conv_markers = pickle.load(f)
    with open(t_path, 'rb') as f:
        t_data = pickle.load(f)
    with open(meta_path, 'rb') as f:
        meta_data = pickle.load(f)
    
    special_lines = np.ones((mean_data.shape[0], mean_data.shape[1]))

    for i in range(len(meta_data['conv_val'])):
        special_lines[i] = meta_data['conv_val'][i] * special_lines[i]

    special_lines, unique_indices = np.unique(special_lines, axis=0, return_index=True)

    if meta_data["special_lines_legend"] is not None:
        special_lines_legend = np.array(meta_data["special_lines_legend"])[unique_indices]
    else:
        special_lines_legend = None
    
    save = False 
    
    if save_root is not None:
        save = True
        
    compare_lines(mean_data, t_data, meta_data["legend"], std_data, 
                  special_lines=special_lines, special_lines_legend=special_lines_legend, 
                  xlabel=meta_data["xlabel"], ylabel=meta_data["ylabel"], 
                  legend_fontsize=meta_data["legend_fontsize"], 
                  colormap=meta_data["colormap"],
                  xlim=meta_data["xlim"], ylim=meta_data["ylim"],
                  title=meta_data["title"], title_fontsize=meta_data["title_fontsize"],
                  markers=conv_markers, save=save, folder=save_root, file_name=save_file_name)