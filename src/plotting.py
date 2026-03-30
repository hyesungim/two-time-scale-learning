import torch
from torch.nn.utils import parameters_to_vector
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
import pickle
from src.config import Config

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12


def get_param_vector(agent):
    return parameters_to_vector(agent.q_net.parameters()).detach().cpu().numpy()


def compute_distance_matrix(agents, metric='euclidean'):
    param_vectors = np.array([get_param_vector(agent) for agent in agents])
    return squareform(pdist(param_vectors, metric=metric))


def visualize_distance_matrix(distance_matrix, generation, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='ward')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    idx = dendro['leaves']
    reordered_dist = distance_matrix[idx, :][:, idx]

    im = ax.imshow(reordered_dist, cmap='viridis', aspect='auto')
    ax.set_title(f'Parameter Distance Matrix - Gen {generation}')
    ax.set_xlabel('Agent ID (Clustered)')
    ax.set_ylabel('Agent ID (Clustered)')
    plt.colorbar(im, ax=ax, label='Distance')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameter_pca(agents, fitness_scores, generation, save_path=None):
    param_vectors = np.array([get_param_vector(agent) for agent in agents])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(param_vectors)
    variance_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=fitness_scores,
                         cmap='RdYlGn', s=100, edgecolors='black', alpha=0.8)

    plt.colorbar(scatter, label='Fitness Score')
    ax.set_title(f'Parameter Space PCA - Gen {generation}')
    ax.set_xlabel(f'PCA 1 ({variance_explained[0] * 100:.1f}% variance)')
    ax.set_ylabel(f'PCA 2 ({variance_explained[1] * 100:.1f}% variance)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_distance_matrix_grid(saved_dist_matrices, target_gens, save_path=None):
    fig, axes = plt.subplots(1, len(target_gens), figsize=(4 * len(target_gens), 3.5))
    if len(target_gens) == 1: axes = [axes]

    for i, gen in enumerate(target_gens):
        ax = axes[i]
        dist_mat = saved_dist_matrices[gen]
        im = ax.imshow(dist_mat, cmap='viridis', aspect='auto')
        ax.set_title(f'Gen {gen}')
        ax.set_xlabel('Agent ID')
        if i == 0:
            ax.set_ylabel('Agent ID')
        else:
            ax.tick_params(left=False, labelleft=False)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Distance')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_parameter_pca_grid(saved_params, saved_fitness, target_gens, save_path=None):
    fig, axes = plt.subplots(1, len(target_gens), figsize=(4 * len(target_gens), 3.5))
    if len(target_gens) == 1: axes = [axes]

    all_fitness = [f for gen_fits in saved_fitness.values() for f in gen_fits]
    vmin, vmax = min(all_fitness), max(all_fitness)

    for i, gen in enumerate(target_gens):
        ax = axes[i]
        param_vectors = saved_params[gen]
        fitness_scores = saved_fitness[gen]

        pca = PCA(n_components=2)
        coords = pca.fit_transform(param_vectors)
        variance_explained = pca.explained_variance_ratio_

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=fitness_scores,
                             cmap='RdYlGn', s=80, edgecolors='black', alpha=0.8,
                             vmin=vmin, vmax=vmax)

        ax.set_title(f'Gen {gen}')
        ax.set_xlabel(f'PC 1 ({variance_explained[0] * 100:.1f}%)')
        if i == 0:
            ax.set_ylabel('PC 2')
        else:
            ax.tick_params(left=False, labelleft=False)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Fitness Score')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_performance(gen_history, fitness_history, raw_reward_history, title=None):
    gens = np.arange(len(fitness_history))
    n_gens = len(gens)
    pop_size = len(fitness_history[0])

    fit_data = np.array(fitness_history)
    raw_data = np.array(raw_reward_history)
    fit_trajectories = fit_data.T

    fit_top5_means, fit_top5_stds = [], []
    raw_top5_means, raw_top5_stds = [], []
    global_maxs = []

    for g in range(n_gens):
        gen_fits = fit_data[g, :]
        top5_idx = np.argsort(gen_fits)[-5:]

        top_5_fits = gen_fits[top5_idx]
        fit_top5_means.append(np.mean(top_5_fits))
        fit_top5_stds.append(np.std(top_5_fits))

        raw_of_top_5 = raw_data[g, top5_idx]
        raw_top5_means.append(np.mean(raw_of_top_5))
        global_maxs.append(np.max(raw_data[g, :]))
        raw_top5_stds.append(np.std(raw_of_top_5))

    fit_top5_means = np.array(fit_top5_means)
    fit_top5_stds = np.array(fit_top5_stds)
    raw_top5_means = np.array(raw_top5_means)
    raw_top5_stds = np.array(raw_top5_stds)
    global_maxs = np.array(global_maxs)

    param_keys = list(gen_history[0][0].keys())
    hparam_trajectories = {k: np.zeros((pop_size, n_gens)) for k in param_keys}
    for g_idx, pop in enumerate(gen_history):
        for a_idx, params in enumerate(pop):
            for k, v in params.items():
                hparam_trajectories[k][a_idx, g_idx] = v

    fig = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(len(param_keys), 2, width_ratios=[1.5, 1])

    ax_main = fig.add_subplot(gs[:, 0])

    for i in range(pop_size):
        ax_main.plot(gens, fit_trajectories[i, :], color='grey', alpha=0.1, linewidth=1)

    ax_main.plot(gens, fit_top5_means, color='#004488', linewidth=1.5, label="Top 5 Mean Fitness")
    ax_main.fill_between(gens, fit_top5_means - fit_top5_stds, fit_top5_means + fit_top5_stds, color='#004488',
                         alpha=0.15)

    ax_main.plot(gens, raw_top5_means, color='#2ca02c', linewidth=1.5, linestyle='-', alpha=0.8,
                 label="Top 5 Mean Episodic Reward")
    ax_main.fill_between(gens, raw_top5_means - raw_top5_stds, raw_top5_means + raw_top5_stds, color='#2ca02c',
                         alpha=0.15)

    ax_main.plot(gens, global_maxs, color="#D55E00", linewidth=2, linestyle="--", label="Best Reward")

    ax_main.set_title("Population Performance", fontsize=14)
    ax_main.set_xlabel("Generations")
    ax_main.set_ylabel("Reward")
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='lower right')

    for i, key in enumerate(param_keys):
        ax = fig.add_subplot(gs[i, 1])
        data = hparam_trajectories[key]
        is_log = "lr" in key.lower() or "learning" in key.lower()

        for a_idx in range(pop_size):
            y_data = np.log10(data[a_idx, :]) if is_log else data[a_idx, :]
            ax.plot(gens, y_data, color='#4ca3dd', alpha=0.2, linewidth=1.5, drawstyle='steps-post')

        if is_log:
            ax.set_title("log$_{10}$(Learning Rate)", fontsize=11)
        elif "epsilon" in key.lower():
            ax.set_title("Greedy Decay Rate", fontsize=11)
        elif "batch" in key.lower():
            ax.set_title("Batch Size", fontsize=11)
        else:
            ax.set_title(key, fontsize=11)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=9)
        if i == len(param_keys) - 1:
            ax.set_xlabel("Generations", fontsize=11)
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    if title is not None:
        plt.savefig(f"{title}.pdf", dpi=300)
    plt.show()


def plot_3d_evolution(filename="hparam_N500_1000steps.pkl", title=None):
    print(f"Loading data from {filename}...")
    try:
        with open(filename, 'rb') as f:
            history_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return

    hparams_array = history_dict["hparams_array"]
    fitness_history = history_dict["fitness"]

    target_gens = [2, len(fitness_history) // 2, -1]
    final_target_gen_name = len(fitness_history)
    titles = ["After 2nd PBT update", f"After {target_gens[1]}th PBT update",
              f"After {final_target_gen_name}th PBT update"]

    all_fitness = np.array(fitness_history)
    global_vmin = all_fitness.min()
    global_vmax = all_fitness.max()

    base_cmap = plt.get_cmap('Blues')
    custom_blues = mcolors.LinearSegmentedColormap.from_list('CustomBlues', base_cmap(np.linspace(0.35, 1.0, 256)))

    fig = plt.figure(figsize=(21, 5.3))
    min_lr, max_lr = -5, -2
    min_eps, max_eps = 500, 5000
    min_bs, max_bs = 32, 128
    sc = None

    for i, gen_idx in enumerate(target_gens):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        current_hparams = hparams_array[gen_idx]
        current_fitness = np.array(fitness_history[gen_idx])

        lr = np.log10(current_hparams[:, 0] + 1e-12)
        eps = current_hparams[:, 1]
        bs = current_hparams[:, 2]

        sort_idx = np.argsort(current_fitness)
        lr = lr[sort_idx]
        eps = eps[sort_idx]
        bs = bs[sort_idx]
        current_fitness = current_fitness[sort_idx]

        sc = ax.scatter(lr, eps, bs, c=current_fitness, cmap=custom_blues, alpha=0.9,
                        s=40, linewidths=0.2, edgecolors='none', vmin=global_vmin, vmax=global_vmax)

        ax.set_xlim(min_lr, max_lr)
        ax.set_ylim(min_eps, max_eps)
        ax.set_zlim(min_bs, max_bs)
        ax.set_xticks(np.linspace(-5, -2, 4))
        ax.set_yticks(np.linspace(1000, 5000, 5))
        ax.set_zticks([32, 64, 96, 128])

        def get_smooth_kde(data, bound_min, bound_max):
            if np.std(data) < 1e-9: data = data + np.random.normal(0, 1e-5, size=len(data))
            kde = gaussian_kde(data)
            x_eval = np.linspace(bound_min, bound_max, 100)
            y_eval = kde(x_eval)
            return x_eval, y_eval / (y_eval.max() + 1e-8)

        lr_eval, lr_norm = get_smooth_kde(lr, min_lr, max_lr)
        eps_eval, eps_norm = get_smooth_kde(eps, min_eps, max_eps)
        bs_eval, bs_norm = get_smooth_kde(bs, min_bs, max_bs)

        color_lr, color_eps, color_bs = '#d62728', '#2ca02c', '#9467bd'
        fill_alpha = 0.2

        z_scaled_lr = min_bs + lr_norm * (max_bs - min_bs) * 0.3
        ax.plot(lr_eval, z_scaled_lr, zs=max_eps, zdir='y', color=color_lr, linewidth=2, alpha=0.8)
        verts_lr = [(lr_eval[0], max_eps, min_bs)] + list(zip(lr_eval, np.full_like(lr_eval, max_eps), z_scaled_lr)) + [
            (lr_eval[-1], max_eps, min_bs)]
        ax.add_collection3d(Poly3DCollection([verts_lr], facecolors=color_lr, alpha=fill_alpha))

        z_scaled_eps = min_bs + eps_norm * (max_bs - min_bs) * 0.3
        ax.plot(eps_eval, z_scaled_eps, zs=min_lr, zdir='x', color=color_eps, linewidth=2, alpha=0.8)
        verts_eps = [(min_lr, eps_eval[0], min_bs)] + list(
            zip(np.full_like(eps_eval, min_lr), eps_eval, z_scaled_eps)) + [(min_lr, eps_eval[-1], min_bs)]
        ax.add_collection3d(Poly3DCollection([verts_eps], facecolors=color_eps, alpha=fill_alpha))

        x_scaled_bs = min_lr + bs_norm * (max_lr - min_lr) * 0.3
        ax.plot(x_scaled_bs, bs_eval, zs=max_eps, zdir='y', color=color_bs, linewidth=2, alpha=0.8)
        verts_bs = [(min_lr, max_eps, bs_eval[0])] + list(zip(x_scaled_bs, np.full_like(bs_eval, max_eps), bs_eval)) + [
            (min_lr, max_eps, bs_eval[-1])]
        ax.add_collection3d(Poly3DCollection([verts_bs], facecolors=color_bs, alpha=fill_alpha))

        ax.set_title(titles[i], fontsize=20)
        ax.set_xlabel('Learning Rate (log$_{10}$)', labelpad=5, fontsize=15)
        ax.set_ylabel('Greedy Policy Decay Rate', labelpad=5, fontsize=15)
        ax.set_zlabel('Batch Size', labelpad=5, fontsize=15)
        ax.tick_params(labelsize=11, pad=2)
        ax.view_init(elev=20, azim=-50)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    cbar_ax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Fitness Score', rotation=270, labelpad=20, fontsize=16)

    if title:
        plt.savefig(f'{title}.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_deque_performance(steps_per_gen, k_values=[1, 2, 5, 10], pop_size=100, metric='raw'):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = sns.color_palette("crest", n_colors=len(k_values))

    for idx, k in enumerate(k_values):
        filename = f"deque{k}_N{pop_size}_{steps_per_gen}steps.pkl"
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Skipping m={k}.")
            continue

        fit_hist = data["fitness"]
        raw_rew_hist = data["raw_rewards"]

        n_gens = len(fit_hist)
        top5_perf_means, pop_perf_means = [], []

        for g in range(n_gens):
            gen_fits = np.array(fit_hist[g])
            top5_idx = np.argsort(gen_fits)[-5:]

            if metric == 'raw':
                all_scores = np.array(raw_rew_hist[g])
                top5_scores = all_scores[top5_idx]
                pop_mean = np.mean(all_scores)
            else:
                top5_scores = gen_fits[top5_idx]
                pop_mean = np.mean(gen_fits)

            top5_perf_means.append(np.mean(top5_scores))
            pop_perf_means.append(pop_mean)

        gen_limit = min(10, n_gens) if steps_per_gen == 500 else n_gens
        gens = np.arange(gen_limit)

        ax.plot(gens, top5_perf_means[:gen_limit], color=colors[idx], linewidth=2, marker='o', markersize=5)
        ax.plot(gens, pop_perf_means[:gen_limit], color=colors[idx], linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_xlabel("PBT updates", fontsize=15)
    ax.set_ylabel("Mean Episodic Reward" if metric == 'raw' else "Mean Fitness Score", fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    style_handles = [
        mlines.Line2D([], [], color='gray', marker='o', markersize=5, linewidth=2, label='Top 5 Mean'),
        mlines.Line2D([], [], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Population Mean')
    ]
    color_handles = [mlines.Line2D([], [], color=colors[i], linewidth=2, label=f"m={k}") for i, k in
                     enumerate(k_values)]
    all_handles = style_handles + color_handles

    ax.legend(handles=all_handles, loc='best', fontsize=13, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"Deque_Size_Performance_{steps_per_gen}steps_N{pop_size}.pdf", dpi=300)
    plt.show()